import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

np.random.seed(0)
tf.random.set_seed(0)

if os.path.exists("Flickr8k.token.txt") and os.path.isdir("Flickr8k_Dataset"):
    print("Dataset: real Flickr8k (first 200 images, first caption each)")
    lines = open("Flickr8k.token.txt").read().strip().split("\n")[::5][:200]
    images, captions = [], []
    for line in lines:
        name, caption = line.split("\t")
        path = os.path.join("Flickr8k_Dataset", name.split("#")[0])
        images.append(keras.utils.img_to_array(keras.utils.load_img(path, target_size=(224, 224))))
        captions.append(caption.lower().replace(".", "").strip())
    images = np.array(images)
else:
    print("FALLBACK: Flickr8k not found. Using 200 CIFAR-10 images with synthetic captions like 'a photo of a cat'")
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    (x, y), _ = keras.datasets.cifar10.load_data()
    images = tf.image.resize(x[:200], (224, 224)).numpy()
    captions = ["a photo of a " + names[i] for i in y[:200].ravel()]

resnet = keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")
features = resnet.predict(keras.applications.resnet50.preprocess_input(images.copy()), batch_size=32, verbose=0)
print("Image features:", features.shape)

words = [["startseq"] + c.split() + ["endseq"] for c in captions]
vocab = {w: i + 1 for i, w in enumerate(sorted({w for c in words for w in c}))}
index_to_word = {i: w for w, i in vocab.items()}
max_len = max(len(c) for c in words)

n_train = 160
feats, seqs, nexts = [], [], []
for f, c in zip(features[:n_train], words[:n_train]):
    ids = [vocab[w] for w in c]
    for i in range(1, len(ids)):
        feats.append(f)
        seqs.append(ids[:i])
        nexts.append(ids[i])
feats = np.array(feats)
seqs = keras.utils.pad_sequences(seqs, maxlen=max_len)
nexts = np.array(nexts)

feature_in = keras.Input((2048,))
f = keras.layers.Dense(128, activation="relu")(feature_in)
seq_in = keras.Input((max_len,))
s = keras.layers.Embedding(len(vocab) + 1, 64, mask_zero=True)(seq_in)
s = keras.layers.LSTM(128)(s)
x = keras.layers.Dense(128, activation="relu")(keras.layers.add([f, s]))
out = keras.layers.Dense(len(vocab) + 1, activation="softmax")(x)
model = keras.Model([feature_in, seq_in], out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit([feats, seqs], nexts, epochs=15, batch_size=64, verbose=0)


def caption_for(feature):
    ids = [vocab["startseq"]]
    for _ in range(max_len):
        seq = keras.utils.pad_sequences([ids], maxlen=max_len)
        word = index_to_word[int(np.argmax(model.predict([feature[None], seq], verbose=0)))]
        if word == "endseq":
            break
        ids.append(vocab[word])
    return " ".join(index_to_word[i] for i in ids[1:])


for i in range(n_train, n_train + 5):
    print("Actual   :", captions[i])
    print("Generated:", caption_for(features[i]))
