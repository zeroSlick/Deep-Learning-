import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"

train_df = pd.read_csv(keras.utils.get_file("ag_train.csv", URL + "train.csv"), header=None).sample(8000, random_state=1)
test_df = pd.read_csv(keras.utils.get_file("ag_test.csv", URL + "test.csv"), header=None).sample(2000, random_state=1)
train_text = (train_df[1] + " " + train_df[2]).values.astype(str)
test_text = (test_df[1] + " " + test_df[2]).values.astype(str)
y_train = train_df[0].values - 1
y_test = test_df[0].values - 1

vectorize = keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=50)
vectorize.adapt(train_text)
x_train = vectorize(train_text).numpy()
x_test = vectorize(test_text).numpy()

inputs = keras.Input((50,))
x = keras.layers.Embedding(10000, 64)(inputs)
attention = keras.layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
x = keras.layers.LayerNormalization()(x + attention)
x = keras.layers.GlobalAveragePooling1D()(x)
outputs = keras.layers.Dense(4, activation="softmax")(x)
transformer = keras.Model(inputs, outputs)

lstm = keras.Sequential([
    keras.layers.Input((50,)),
    keras.layers.Embedding(10000, 64),
    keras.layers.LSTM(64),
    keras.layers.Dense(4, activation="softmax"),
])

for name, model in [("Transformer", transformer), ("LSTM", lstm)]:
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start = time.time()
    model.fit(x_train, y_train, epochs=4, batch_size=64, verbose=0)
    seconds = time.time() - start
    acc = model.evaluate(x_test, y_test, verbose=0)[1]
    print(name, "- test accuracy:", round(acc, 4), "training time:", round(seconds, 1), "s")
