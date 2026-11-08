import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from myLLM import CustomLLM


def text_to_ids(text_tokens):
    return [token2idx[t] for t in text_tokens if t in token2idx]


# --- 1. Load token mappings ---
with open("token2idx.json", "r") as f:
    token2idx = json.load(f)
with open("idx2token.json", "r") as f:
    idx2token = json.load(f)

vocab_size = len(token2idx)
max_len = 100

# --- 2. Prepare the same dataset using Nietzsche text ---
dataset_path = keras.utils.get_file(
    'nietzche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

with open(dataset_path, 'r', encoding='utf-8') as f:
    text = f.read()

tokens = text.split()
ids = text_to_ids(tokens)

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
model = CustomLLM(vocab_size, max_len, embed_dim=256, num_heads=8, ff_dim=512)

# --- 4. Compile ---
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'], jit_compile=False)

model.summary()

# --- 5. Train ---

# enable tensorboard logs
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")

print("Starting training...")
#model.fit(X, Y, batch_size=32, epochs=2, [tensorboard_callback])
model.fit(X, Y, batch_size=32, epochs=2)

model.summary()

keras.utils.plot_model(
    model,
    to_file="simple_llm_model.png",  # Save the plot to a file
    show_shapes=True,  # Display shape information
    show_layer_names=True,  # Display layer names
    show_layer_activations=True,  # Display activation functions
    rankdir="TB"  # Orientation: TB=Top-to-Bottom, LR=Left-to-Right
)

# --- 6. Generate text ---
print("\nGenerating text...")
# Start with a prefix sequence from the training data
start_seq = X[0].tolist() 
generated = start_seq.copy()


# Step A: Initialize the cache with the prompt sequence
# Convert prompt sequence to array layout: (batch_size=1, seq_len)
current_input = np.array([start_seq])

# Call model directly using the return_cache option
logits, cache = model(current_input, training=False, return_cache=True)

# Extract the prediction for the absolute last token of the prompt sequence
next_token_logits = logits[0, -1, :]
next_idx = int(tf.argmax(next_token_logits).numpy())
generated.append(next_idx)


# Step B: Auto-regressive loop using the cache step-by-step
for _ in range(50):
    # We ONLY pass the single latest token index! Sequence length = 1
    current_input = np.array([[next_idx]])

    # Pass the running history state ('cache') back into the model
    logits, cache = model(current_input, training=False, cache=cache, return_cache=True)

    # Because input shape sequence dimension is 1, index the unique token position
    next_token_logits = logits[0, 0, :]
    next_idx = int(tf.argmax(next_token_logits).numpy())
    generated.append(next_idx)

# Convert back to readable text tokens
generated_text = " ".join([idx2token[str(i)] for i in generated if str(i) in idx2token])
print(f"\nGenerated text output:\n{generated_text}")

