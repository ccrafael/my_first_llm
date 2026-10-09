import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from myLLM import CustomLLM, LLMDecoderBlock

# --- 1. Load token mappings ---
with open("char2idx.json", "r") as f:
    char2idx = json.load(f)
with open("idx2char.json", "r") as f:
    idx2char = json.load(f)

vocab_size = len(char2idx)
max_len = 100

# --- 2. Prepare the same corpus used for training ---
corpus = """
Artificial intelligence is transforming the world. 
Machine learning models can understand and generate human-like text. 
Transformers are a powerful architecture for processing sequences. 
This model will learn patterns from the text data.
"""

def text_to_ids(text):
    return [char2idx[c] for c in text]

ids = text_to_ids(corpus)

seq_length = 50
xs = []
ys = []
for i in range(0, len(ids) - seq_length, 1):
    xs.append(ids[i:i + seq_length])
    ys.append(ids[i + 1:i + seq_length + 1])

X = np.array(xs)
Y = np.array(ys)

# --- 3. Initialize the model ---
print("Building model...")
model = CustomLLM(vocab_size, max_len, embed_dim=256, num_heads=4, ff_dim=512)

# --- 4. Compile ---
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'], jit_compile=False)

# --- 5. Train ---
print("Starting training...")
model.fit(X, Y, batch_size=32, epochs=3)

# --- 6. Generate text ---
print("\nGenerating text...")
# Start with the first valid token (not padding)
start_token = char2idx.get(" ", 1)  # start with space or fallback
generated = [start_token]

for _ in range(50):
    # Create input sequence: pad to max_len with the generated tokens on the right
    padded = [0] * (max_len - len(generated)) + generated
    pred = model.predict(np.array([padded]), verbose=0)[0]  # shape: (max_len, vocab_size)
    # Get logits for the last position
    next_logits = pred[-1]  # shape: (vocab_size,)
    next_idx = int(tf.argmax(next_logits).numpy())
    generated.append(next_idx)

# Convert back to text - idx2char keys are strings
generated_text = "".join([idx2char[str(i)] for i in generated if str(i) in idx2char])
print(f"Generated text: {generated_text}")
