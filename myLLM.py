import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class LLMDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        
        # 1. Masked Self-Attention (The ONLY attention layer now!)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        
        # 2. Feed-Forward Network
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(rate)
  
    def call(self, inputs, training=False, mask=None, cache=None):
        # Step 1: Masked Self Attention with optional kv cache
        if cache is not None and len(cache) > 0:
            # Cache contains (key, value) from previous tokens
            key, value = cache
            new_attn_out = self.att(
                query=inputs,
                key=key,
                value=value,
                use_causal_mask=True,
            )
        else:
            # Full attention computation (training or first token)
            new_attn_out = self.att(
                query=inputs, 
                value=inputs, 
                key=inputs, 
                use_causal_mask=True, 
                attention_mask=mask
            )
        new_attn_out = self.dropout1(new_attn_out, training=training)
        # Residual connection + Normalization
        out1 = self.layernorm1(inputs + new_attn_out) 
        
        # Step 2: Feed Forward Network
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        # Final Residual + Normalization
        result = self.layernorm2(out1 + ffn_out)
        
        # Compute key and value for caching
        cache_key = new_attn_out
        cache_value = new_attn_out
        
        # Update cache
        if cache is not None:
            # Append new key/value to existing cache
            updated_cache = cache + (cache_key, cache_value)
        else:
            # Initialize cache with just this step's key and value
            updated_cache = (cache_key, cache_value)
        
        return result, updated_cache


class CustomLLM(keras.Model):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim):
        super().__init__()
        # The "Eyes": Turn token IDs into vectors + Add Positional numbers
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

        # The "Brain": Our custom block
        self.transformer_block = LLMDecoderBlock(embed_dim, num_heads, ff_dim)

        # The "Mouth": Convert the final vector back into a vocabulary prediction
        self.final_layer = layers.Dense(vocab_size)

    def call(self, inputs, training=False, cache=None):
        # 1. Get word embeddings
        # Handle both (batch,) and (batch, seq_len) input shapes
        if len(inputs.shape) == 2:
            # inputs already has sequence dimension: (batch, seq_len)
            x = self.token_emb(inputs)
            # x will be (batch, seq_len, embed_dim)
        else:
            # inputs is (batch,) - treat as single token or sequence
            # Embedding will give (batch, embed_dim)
            x = self.token_emb(inputs)
            # Expand to (batch, 1, embed_dim) - treat as sequence of length 1
            x = tf.expand_dims(x, axis=1)
        
        # 2. Add positional embeddings
        # Get sequence length from the actual tensor dimension
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions_emb = self.pos_emb(positions)
        
        # Ensure positions_emb is 3D: (1, seq_len, embed) if needed
        if len(positions_emb.shape) == 2:
            positions_emb = tf.expand_dims(positions_emb, axis=0)
        
        x = x + positions_emb
        
        # 3. Pass through our Decoder block with optional cache
        x, cache = self.transformer_block(x, training=training, cache=cache)
        
        # 4. Output the raw vocabulary scores (logits)
        logits = self.final_layer(x)
        
        return logits


