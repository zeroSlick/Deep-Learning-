import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

rng = np.random.default_rng(0)
tf.random.set_seed(0)

if os.path.exists("events.csv"):
    print("Dataset: real Retailrocket (events.csv, first 200000 rows)")
    df = pd.read_csv("events.csv", usecols=["visitorid", "itemid"], nrows=200000).drop_duplicates()
    df = df[df.itemid.isin(df.itemid.value_counts().index[:500])]
    df = df[df.groupby("visitorid").itemid.transform("count") >= 3]
    pairs = np.column_stack([df.visitorid.astype("category").cat.codes, df.itemid.astype("category").cat.codes])
else:
    print("FALLBACK: events.csv not found. Using synthetic interactions (5 user groups, 5 item groups)")
    pairs = []
    for user in range(200):
        group = user % 5
        items = list(rng.choice(20, 12, replace=False) + group * 20) + list(rng.choice(100, 3))
        pairs += [(user, i) for i in set(items)]
    pairs = np.array(pairs)

n_users, n_items = int(pairs[:, 0].max()) + 1, int(pairs[:, 1].max()) + 1
is_test = rng.random(len(pairs)) < 0.2
train_pairs, test_pairs = pairs[~is_test], pairs[is_test]

negatives = np.column_stack([rng.choice(train_pairs[:, 0], len(train_pairs) * 4), rng.integers(0, n_items, len(train_pairs) * 4)])
users = np.concatenate([train_pairs[:, 0], negatives[:, 0]])
items = np.concatenate([train_pairs[:, 1], negatives[:, 1]])
labels = np.concatenate([np.ones(len(train_pairs)), np.zeros(len(negatives))])

user_in = keras.Input((1,))
item_in = keras.Input((1,))
user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users, 16, name="user")(user_in))
item_vec = keras.layers.Flatten()(keras.layers.Embedding(n_items, 16, name="item")(item_in))
out = keras.layers.Activation("sigmoid")(keras.layers.Dot(axes=1)([user_vec, item_vec]))
model = keras.Model([user_in, item_in], out)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit([users, items], labels, epochs=30, batch_size=256, verbose=0)

scores = model.get_layer("user").get_weights()[0] @ model.get_layer("item").get_weights()[0].T
scores[train_pairs[:, 0], train_pairs[:, 1]] = -np.inf

K = 10
top_k = np.argsort(-scores, axis=1)[:, :K]
precisions, recalls = [], []
for user in range(n_users):
    held_out = set(test_pairs[test_pairs[:, 0] == user, 1])
    if held_out:
        hits = len(held_out & set(top_k[user]))
        precisions.append(hits / K)
        recalls.append(hits / len(held_out))

print("Top", K, "recommendations for user 0:", top_k[0])
print("Precision@%d: %.4f" % (K, np.mean(precisions)))
print("Recall@%d: %.4f" % (K, np.mean(recalls)))
