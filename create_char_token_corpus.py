import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- 1. Create a small corpus of data ---
corpus = """
Artificial intelligence is transforming the world. 
Machine learning models can understand and generate human-like text. 
Transformers are a powerful architecture for processing sequences. 
This model will learn patterns from the text data.
"""

# --- 2. Simple character-level tokenizer ---
chars = sorted(list(set(corpus)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

# --- 3. Prepare sequence data ---
def text_to_ids(text):
    return [char2idx[c] for c in text]

ids = text_to_ids(corpus)

# Create input-output pairs for training
# X: input sequence, Y: next token (shifted by 1)
seq_length = 50  # length of each input sequence

xs = []
ys = []
for i in range(0, len(ids) - seq_length, 1):
    xs.append(ids[i:i + seq_length])
    ys.append(ids[i + 1:i + seq_length + 1])

X = np.array(xs)
Y = np.array(ys)

print(f"Created {len(X)} training sequences of length {seq_length}")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# --- 4. Save token mappings for later use ---
import json
with open("char2idx.json", "w") as f:
    json.dump(char2idx, f)
with open("idx2char.json", "w") as f:
    json.dump(idx2char, f)

print("Token mappings saved.")