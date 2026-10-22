import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_recall_fscore_support

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

print("FALLBACK: CoNLL-2003 not used. Using a small synthetic labelled NER dataset.")
people = ["john", "mary", "alice", "bob", "david", "sara"]
orgs = ["google", "microsoft", "amazon", "apple", "tesla"]
places = ["london", "paris", "delhi", "tokyo", "berlin"]
tags = ["O", "PER", "ORG", "LOC"]

sentences, labels = [], []
for _ in range(400):
    p, o, l = random.choice(people), random.choice(orgs), random.choice(places)
    kind = random.randint(0, 3)
    if kind == 0:
        s, t = [p, "works", "at", o, "in", l], ["PER", "O", "O", "ORG", "O", "LOC"]
    elif kind == 1:
        s, t = [p, "visited", l, "last", "week"], ["PER", "O", "LOC", "O", "O"]
    elif kind == 2:
        s, t = [o, "opened", "an", "office", "in", l], ["ORG", "O", "O", "O", "O", "LOC"]
    else:
        s, t = [p, "joined", o], ["PER", "O", "ORG"]
    sentences.append(s)
    labels.append([tags.index(x) for x in t])

vocab = {w: i + 1 for i, w in enumerate(sorted({w for s in sentences for w in s}))}
x = keras.utils.pad_sequences([[vocab[w] for w in s] for s in sentences], maxlen=6, padding="post")
y = keras.utils.pad_sequences(labels, maxlen=6, padding="post")
x_train, x_test, y_train, y_test = x[:320], x[320:], y[:320], y[320:]

model = keras.Sequential([
    keras.layers.Input((6,)),
    keras.layers.Embedding(len(vocab) + 1, 32, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True)),
    keras.layers.Dense(4, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(x_train, y_train, epochs=15, batch_size=16, verbose=0)

pred = np.argmax(model.predict(x_test, verbose=0), axis=-1)
real = x_test != 0
precision, recall, f1, _ = precision_recall_fscore_support(y_test[real], pred[real], labels=[1, 2, 3], average="micro")
print("Entity precision:", round(precision, 4), "recall:", round(recall, 4), "F1:", round(f1, 4))

words = {i: w for w, i in vocab.items()}
for i in range(3):
    print([(words[w], tags[t]) for w, t in zip(x_test[i], pred[i]) if w != 0])
