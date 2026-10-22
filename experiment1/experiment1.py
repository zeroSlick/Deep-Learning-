import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
print("Add:\n", (a + b).numpy())
print("Matmul:\n", tf.matmul(a, b).numpy())
print("Mean:", tf.reduce_mean(a).numpy())

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:10000] / 255.0
y_train = y_train[:10000]
x_test = x_test / 255.0

perceptron = keras.Sequential([
    keras.layers.Input((28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation="softmax"),
])

mlp = keras.Sequential([
    keras.layers.Input((28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

for name, model in [("Perceptron", perceptron), ("MLP", mlp)]:
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(name, "test accuracy:", round(acc, 4))
