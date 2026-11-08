import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AttentionHead(layers.Layer):
    def __init__(self, head_dim):
        super().__init__()
        self.q = layers.Dense(head_dim)
        self.k = layers.Dense(head_dim)
        self.v = layers.Dense(head_dim)


    def call(self, hidden_state, mask=None, cache=None, use_causal_mask=False):
        # 1. Calculate current Query, Key, and Value
        query = self.q(hidden_state)
        key = self.k(hidden_state)
        value = self.v(hidden_state)

        if cache is not None:
            prev_key, prev_value = cache

            # Concatenate past keys/values with the new ones along the sequence axis (axis=1)
            key = keras.ops.concatenate([prev_key, key], axis=1)
            value = keras.ops.concatenate([prev_value, value], axis=1)

        # Save the updated key and value to return back to the model
        updated_cache = (key, value)


        # 
        # Scaled dot-product attention!!!
        #
        dim_k = tf.cast(tf.shape(query)[-1], tf.float32)
        
        # query shape: (batch, seq_len_q, head_dim)
        # key shape:   (batch, seq_len_k, head_dim)
        # scores shape: (batch, seq_len_q, seq_len_k)
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(dim_k)

        # Apply Masking
        # Causal mask for training / auto-regressive generation setup
        if use_causal_mask:
            seq_len_q = tf.shape(query)[1]
            seq_len_k = tf.shape(key)[1]
            # Create a matrix of 1s and 0s where 1 means "keep" and 0 means "mask out"
            # I look back from the end of the key sequence length
            causal_mask = tf.linalg.band_part(tf.ones((seq_len_q, seq_len_k)), -1, 0)
            # Expand dimensions to broadcast over the batch: (1, seq_len_q, seq_len_k)
            causal_mask = tf.expand_dims(causal_mask, axis=0)

            # Apply the mask by adding a very large negative number to masked-out spots
            # When softmax hits -1e9, it turns into exactly 0.0 probability
            scores = tf.where(causal_mask == 1, scores, -1e9)

        # Additional padding mask if passed explicitly from tokenization step
        if mask is not None:
            # Reshape padding mask to match scores matrix layout if necessary
            scores = tf.where(mask, scores, -1e9)

        # Softmax and Value weighting
        weights = keras.activations.softmax(scores, axis=-1)
        output = tf.matmul(weights, value)

        return output, updated_cache


class MultiHeadAttention(layers.Layer):
    def __init__(self, num_attention_heads, hidden_size):
        super().__init__()
        embed_dim = hidden_size
        self.num_heads = num_attention_heads
        head_dim = embed_dim // self.num_heads
        
        self.heads = [AttentionHead(head_dim) for _ in range(self.num_heads)]
        self.output_linear = layers.Dense(embed_dim)

    def call(self, hidden_state, mask=None, cache=None, use_causal_mask=False):
        head_outputs = []
        updated_caches = []

        # Route inputs to individual heads alongside their specific cache slice
        for i, head in enumerate(self.heads):
            head_cache = cache[i] if cache is not None else None
            
            out, u_cache = head(
                hidden_state, 
                mask=mask, 
                cache=head_cache, 
                use_causal_mask=use_causal_mask
            )
            head_outputs.append(out)
            updated_caches.append(u_cache)

        # Standard MHA combine sequence: Concatenate -> Project Linear
        x = keras.ops.concatenate(head_outputs, axis=-1)
        x = self.output_linear(x)
        
        # Return calculated sequence representations + the updated cache structure
        return x, tuple(updated_caches)


class LLMDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        
        # 1. Masked Self-Attention (The ONLY attention layer now!)
        #self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att = MultiHeadAttention(num_attention_heads=num_heads, hidden_size=embed_dim)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        
        # 2. Feed-Forward Network
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(rate)

  
    def call(self, inputs, training=False, mask=None, cache=None, use_causal_mask=True):
        # Pass inputs through MultiHeadAttention along with cache parameters
        new_attn_out, updated_cache = self.att(
            inputs, 
            mask=mask, 
            cache=cache, 
            use_causal_mask=use_causal_mask
        )
        
        new_attn_out = self.dropout1(new_attn_out, training=training)
        out1 = self.layernorm1(inputs + new_attn_out)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        result = self.layernorm2(out1 + ffn_out)

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


    def call(self, inputs, training=False, mask=None, cache=None, use_causal_mask=True, return_cache=False):
        # Get word embeddings
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
        
        # Add positional embeddings
        # Get sequence length from the actual tensor dimension
        seq_len = tf.shape(x)[1]

        if cache is not None:
            # Extract historical steps length from an active cache item (e.g. Head 0 Key component)
            past_seq_len = tf.shape(cache[0][0])[1]
            positions = tf.range(start=past_seq_len, limit=past_seq_len + seq_len, delta=1)
        else:
            positions = tf.range(start=0, limit=seq_len, delta=1)

        positions_emb = self.pos_emb(positions)
        if len(positions_emb.shape) == 2:
            positions_emb = tf.expand_dims(positions_emb, axis=0)

        x = x + positions_emb

        # Core model pass-through execution
        x, updated_cache = self.transformer_block(
            x,
            training=training,
            mask=mask,
            cache=cache,
            use_causal_mask=use_causal_mask
        )

        logits = self.final_layer(x)
        if return_cache:
            return logits, updated_cache

        return logits

