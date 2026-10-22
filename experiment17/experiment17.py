import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

np.random.seed(0)
tf.random.set_seed(0)

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
noisy_train = np.clip(x_train + 0.4 * np.random.normal(size=x_train.shape), 0, 1)
noisy_test = np.clip(x_test + 0.4 * np.random.normal(size=x_test.shape), 0, 1)

model = keras.Sequential([
    keras.layers.Input((28, 28, 1)),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same"),
])
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(noisy_train, x_train, epochs=5, batch_size=128, verbose=2)

denoised = model.predict(noisy_test[:1000], verbose=0)
print("MSE noisy vs clean:", np.mean((noisy_test[:1000] - x_test[:1000]) ** 2))
print("MSE denoised vs clean:", np.mean((denoised - x_test[:1000]) ** 2))
print("PSNR noisy:", tf.reduce_mean(tf.image.psnr(noisy_test[:1000], x_test[:1000], 1.0)).numpy())
print("PSNR denoised:", tf.reduce_mean(tf.image.psnr(denoised, x_test[:1000], 1.0)).numpy())

plt.figure(figsize=(10, 4))
for i in range(8):
    for row, images in enumerate([noisy_test, x_test, denoised]):
        plt.subplot(3, 8, row * 8 + i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
plt.suptitle("Top: noisy, Middle: clean, Bottom: denoised")
plt.show()
