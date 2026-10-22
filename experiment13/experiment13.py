import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train[:10000] / 255.0
y_train = y_train[:10000]
x_test = x_test[:2000] / 255.0
y_test = y_test[:2000]

for activation in ["sigmoid", "tanh", "relu"]:
    for layers, neurons in [(1, 32), (2, 64)]:
        model = keras.Sequential([keras.layers.Input((28, 28)), keras.layers.Flatten()])
        for _ in range(layers):
            model.add(keras.layers.Dense(neurons, activation=activation))
        model.add(keras.layers.Dense(10, activation="softmax"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
        acc = model.evaluate(x_test, y_test, verbose=0)[1]
        print(activation, "| hidden layers:", layers, "| neurons:", neurons, "| accuracy:", round(acc, 4))
