import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        
        # 1. Masked Self-Attention (Decoder looking at itself)
        self.att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        
        # 2. Cross-Attention (Decoder looking at the Encoder's KV output)
        self.att2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(rate)
        
        # 3. Feed-Forward Network (Processing the new context)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout3 = layers.Dropout(rate)

    def call(self, inputs, encoder_out, training=False, mask=None):
        # Step 1: Masked Self Attention
        # Notice we use `use_causal_mask=True` so it can't look into the future!
        attn_out_1 = self.att1(
            query=inputs, 
            value=inputs, 
            key=inputs, 
            use_causal_mask=True, 
            attention_mask=mask
        )
        attn_out_1 = self.dropout1(attn_out_1, training=training)
        # Residual connection + Normalization
        out1 = self.layernorm1(inputs + attn_out_1) 
        
        # Step 2: Cross Attention
        # Query comes from Decoder (out1). Key and Value come from Encoder (encoder_out)
        attn_out_2 = self.att2(
            query=out1, 
            value=encoder_out, 
            key=encoder_out, 
            attention_mask=None # You can add padding masks here if needed
        )
        attn_out_2 = self.dropout2(attn_out_2, training=training)
        # Residual connection + Normalization
        out2 = self.layernorm2(out1 + attn_out_2) 
        
        # Step 3: Feed Forward Network
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        # Final Residual + Normalization
        return self.layernorm3(out2 + ffn_out)



# Create one Decoder Block
decoder_layer = TransformerDecoderBlock(embed_dim=256, num_heads=8, ff_dim=512)

# Feed it some dummy data (Batch size: 32, Sequence Length: 10, Embedding Dim: 256)
dummy_decoder_inputs = tf.random.normal((32, 10, 256))
dummy_encoder_outputs = tf.random.normal((32, 20, 256)) # Encoder read a 20-word sentence

# Run the data through!
output = decoder_layer(dummy_decoder_inputs, dummy_encoder_outputs)

print(output.shape) 
# Output: (32, 10, 256)


