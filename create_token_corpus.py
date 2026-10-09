import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Download Nietzsche dataset ---
keras.utils.get_file('nietzche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
path = "/home/rafael/.keras/datasets/nietzche.txt"

# --- Read and prepare text ---
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text)} characters")

# --- Tokenize using word-level (split by whitespace) ---
tokens = text.split()
vocab_size = len(set(tokens))
token2idx = {token: i for i, token in enumerate(set(tokens))}
idx2token = {i: token for i, token in enumerate(set(tokens))}

print(f"Vocabulary size: {vocab_size} unique tokens")
print(f"Sample tokens: {list(token2idx.keys())[:20]}")

# --- Prepare sequence data ---
seq_length = 50

xs = []
ys = []
for i in range(0, len(tokens) - seq_length, 1):
    xs.append([token2idx[tokens[i+j]] for j in range(seq_length)])
    ys.append(token2idx[tokens[i + seq_length]])

X = np.array(xs)
Y = np.array(ys)

print(f"Created {len(X)} training sequences of length {seq_length}")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# --- Save token mappings ---
import json
with open("token2idx.json", "w") as f:
    json.dump(token2idx, f)
with open("idx2token.json", "w") as f:
    json.dump(idx2token, f)

print("Token mappings saved.")