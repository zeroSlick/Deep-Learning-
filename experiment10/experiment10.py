import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

print("FALLBACK: Sentiment140 not used. Using a small synthetic set of short posts.")
things = ["movie", "phone", "game", "food", "song", "place"]
posts, labels = [], []
for t in things:
    for w in ["love", "enjoy", "adore"]:
        posts.append("i " + w + " this " + t)
        labels.append(1)
    for w in ["hate", "dislike", "despise"]:
        posts.append("i " + w + " this " + t)
        labels.append(0)
    for w in ["great", "amazing", "wonderful"]:
        posts.append("this " + t + " is " + w)
        labels.append(1)
    for w in ["terrible", "awful", "horrible"]:
        posts.append("this " + t + " is " + w)
        labels.append(0)

order = np.random.permutation(len(posts))
posts = np.array(posts)[order]
labels = np.array(labels)[order]
split = int(0.8 * len(posts))

vectorize = keras.layers.TextVectorization(max_tokens=200, output_sequence_length=6)
vectorize.adapt(posts[:split])
x = vectorize(posts).numpy()

model = keras.Sequential([
    keras.layers.Input((6,)),
    keras.layers.Embedding(200, 16),
    keras.layers.LSTM(16),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x[:split], labels[:split], epochs=30, batch_size=8, verbose=0)
print("Test accuracy:", round(model.evaluate(x[split:], labels[split:], verbose=0)[1], 4))

graph = nx.erdos_renyi_graph(15, 0.25, seed=1)
user_posts = random.sample(list(posts), 15)
scores = model.predict(vectorize(np.array(user_posts)).numpy(), verbose=0).ravel()
colors = ["green" if s > 0.5 else "red" for s in scores]
for node in graph.nodes:
    print("User", node, "-", user_posts[node], "->", "positive" if scores[node] > 0.5 else "negative")

nx.draw(graph, node_color=colors, with_labels=True, font_color="white")
plt.title("User graph (green = positive, red = negative)")
plt.show()
