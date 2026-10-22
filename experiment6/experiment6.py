import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

autoencoder = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(784, activation="sigmoid"),
])
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test), verbose=2)

decoded = autoencoder.predict(x_test, verbose=0)
print("Reconstruction MSE:", np.mean((x_test - decoded) ** 2))

plt.figure(figsize=(10, 3))
for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.subplot(2, 8, i + 9)
    plt.imshow(decoded[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.suptitle("Top: original, Bottom: reconstructed")
plt.show()
