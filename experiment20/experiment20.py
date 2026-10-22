import numpy as np
import tensorflow as tf
from tensorflow import keras

np.random.seed(0)
tf.random.set_seed(0)

try:
    path = keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
    text = open(path, encoding="utf-8").read()[:30000].lower()
    print("Corpus: real Shakespeare text (first 30000 characters)")
except Exception:
    print("FALLBACK: Shakespeare download failed, using a tiny built-in text")
    text = "to be, or not to be, that is the question: whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune, " * 20

chars = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(chars)}
seq_len = 40

x = np.array([[char_to_index[c] for c in text[i:i + seq_len]] for i in range(0, len(text) - seq_len, 3)])
y = np.array([char_to_index[text[i + seq_len]] for i in range(0, len(text) - seq_len, 3)])

model = keras.Sequential([
    keras.layers.Input((seq_len,)),
    keras.layers.Embedding(len(chars), 32),
    keras.layers.LSTM(128),
    keras.layers.Dense(len(chars), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(x, y, epochs=25, batch_size=128, verbose=2)

for temperature in [0.5, 1.0]:
    generated = text[:seq_len]
    for _ in range(200):
        seq = np.array([[char_to_index[c] for c in generated[-seq_len:]]])
        probs = model.predict(seq, verbose=0)[0].astype("float64")
        probs = np.exp(np.log(probs + 1e-8) / temperature)
        probs /= probs.sum()
        generated += chars[np.random.choice(len(chars), p=probs)]
    print("\nTemperature", temperature, "\n" + generated)
