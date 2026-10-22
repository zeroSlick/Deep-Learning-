import os
import re
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

random.seed(0)
tf.random.set_seed(0)

try:
    path = keras.utils.get_file("fra-eng.zip", "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip", extract=True)
    lines = open(os.path.join(path if os.path.isdir(path) else os.path.dirname(path), "fra.txt"), encoding="utf-8").read().split("\n")[:30000]
    print("Dataset: real English-French (ManyThings), 3000 random pairs from the first 30000")
except Exception:
    print("FALLBACK: ManyThings download failed, using a tiny built-in set")
    base = ["i am happy\tje suis heureux", "i am tired\tje suis fatigue", "he is tall\til est grand", "she is kind\telle est gentille",
            "we are late\tnous sommes en retard", "you are nice\ttu es gentil", "i like tea\tj aime le the", "he likes cats\til aime les chats"]
    lines = base * 10

pairs = []
lines = [l for l in lines if "\t" in l]
for line in random.sample(lines, min(3000, len(lines))):
    eng, fre = line.split("\t")[:2]
    pairs.append((re.sub(r"[^a-z ]", "", eng.lower().replace("'", " ")), "<s> " + re.sub(r"[^a-zéèêàçùôîâ ]", "", fre.lower().replace("'", " ")) + " </s>"))

split = int(0.9 * len(pairs))
eng_vocab = {w: i + 1 for i, w in enumerate(sorted({w for e, f in pairs for w in e.split()}))}
fre_vocab = {w: i + 1 for i, w in enumerate(sorted({w for e, f in pairs for w in f.split()}))}
fre_words = {i: w for w, i in fre_vocab.items()}
enc_len = max(len(e.split()) for e, f in pairs)
dec_len = max(len(f.split()) for e, f in pairs) - 1

encoder_x = keras.utils.pad_sequences([[eng_vocab[w] for w in e.split()] for e, f in pairs], maxlen=enc_len, padding="post")
decoder_x = keras.utils.pad_sequences([[fre_vocab[w] for w in f.split()][:-1] for e, f in pairs], maxlen=dec_len, padding="post")
decoder_y = keras.utils.pad_sequences([[fre_vocab[w] for w in f.split()][1:] for e, f in pairs], maxlen=dec_len, padding="post")

enc_in = keras.Input((enc_len,))
enc_emb = keras.layers.Embedding(len(eng_vocab) + 1, 64, mask_zero=True)(enc_in)
enc_out, h, c = keras.layers.LSTM(128, return_sequences=True, return_state=True)(enc_emb)

dec_in = keras.Input((dec_len,))
dec_emb = keras.layers.Embedding(len(fre_vocab) + 1, 64, mask_zero=True)(dec_in)
dec_out = keras.layers.LSTM(128, return_sequences=True)(dec_emb, initial_state=[h, c])
context = keras.layers.Attention()([dec_out, enc_out])
out = keras.layers.Dense(len(fre_vocab) + 1, activation="softmax")(keras.layers.Concatenate()([dec_out, context]))

model = keras.Model([enc_in, dec_in], out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit([encoder_x[:split], decoder_x[:split]], decoder_y[:split], epochs=25, batch_size=64, verbose=2)


def translate(enc_batch):
    dec = np.zeros((len(enc_batch), dec_len), dtype="int32")
    dec[:, 0] = fre_vocab["<s>"]
    for t in range(dec_len - 1):
        dec[:, t + 1] = np.argmax(model([enc_batch, dec], training=False).numpy()[:, t], axis=-1)
    results = []
    for row in dec:
        words = []
        for i in row[1:]:
            if i == 0 or fre_words[i] == "</s>":
                break
            words.append(fre_words[i])
        results.append(words)
    return results


predicted = translate(encoder_x[split:])
references = [[pairs[split + i][1].split()[1:-1]] for i in range(len(predicted))]
print("BLEU on held-out pairs:", round(corpus_bleu(references, predicted, smoothing_function=SmoothingFunction().method1), 4))

for i in range(5):
    print("English:", pairs[split + i][0], "| Reference:", " ".join(references[i][0]), "| Predicted:", " ".join(predicted[i]))
