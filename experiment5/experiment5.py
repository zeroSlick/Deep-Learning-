import time
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=5000)
x_train = keras.utils.pad_sequences(x_train[:5000], maxlen=100)
y_train = y_train[:5000]
x_test = keras.utils.pad_sequences(x_test[:2000], maxlen=100)
y_test = y_test[:2000]

for name, layer in [("SimpleRNN", keras.layers.SimpleRNN(32)), ("LSTM", keras.layers.LSTM(32)), ("GRU", keras.layers.GRU(32))]:
    model = keras.Sequential([
        keras.layers.Input((100,)),
        keras.layers.Embedding(5000, 32),
        layer,
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    start = time.time()
    model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)
    seconds = time.time() - start
    acc = model.evaluate(x_test, y_test, verbose=0)[1]
    print(name, "- test accuracy:", round(acc, 4), "training time:", round(seconds, 1), "s")
