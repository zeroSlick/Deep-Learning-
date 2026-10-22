import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

(x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
data = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(128)

inputs = keras.Input((784,))
h = keras.layers.Dense(256, activation="relu")(inputs)
z_mean = keras.layers.Dense(2)(h)
z_log_var = keras.layers.Dense(2)(h)
encoder = keras.Model(inputs, [z_mean, z_log_var])

decoder = keras.Sequential([
    keras.layers.Input((2,)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(784, activation="sigmoid"),
])

optimizer = keras.optimizers.Adam(0.001)


@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        mean, log_var = encoder(x)
        z = mean + tf.exp(0.5 * log_var) * tf.random.normal(tf.shape(mean))
        recon = decoder(z)
        recon_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(x[:, :, None], recon[:, :, None]), axis=1))
        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1))
        loss = recon_loss + kl_loss
    variables = encoder.trainable_variables + decoder.trainable_variables
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
    return loss


for epoch in range(10):
    for x in data:
        loss = train_step(x)
    print("Epoch", epoch + 1, "- loss:", round(float(loss), 2))

mean, _ = encoder(x_test[:5000])
plt.scatter(mean[:, 0], mean[:, 1], c=y_test[:5000], cmap="tab10", s=3)
plt.colorbar()
plt.title("Latent space")
plt.show()

images = decoder(tf.random.normal((16, 2))).numpy().reshape(-1, 28, 28)
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.axis("off")
plt.suptitle("Digits generated from random latent samples")
plt.show()
