import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = (x_train.reshape(-1, 784).astype("float32") - 127.5) / 127.5
data = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(128, drop_remainder=True)

generator = keras.Sequential([
    keras.layers.Input((100,)),
    keras.layers.Dense(256),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(512),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dense(784, activation="tanh"),
])
discriminator = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(256),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation="sigmoid"),
])

bce = keras.losses.BinaryCrossentropy()
g_opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
d_opt = keras.optimizers.Adam(0.0001, beta_1=0.5)


@tf.function
def train_step(real):
    noise = tf.random.normal((128, 100))
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake = generator(noise)
        real_out = discriminator(real, training=True)
        fake_out = discriminator(fake, training=True)
        g_loss = bce(tf.ones_like(fake_out), fake_out)
        d_loss = bce(0.9 * tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)
    g_opt.apply_gradients(zip(g_tape.gradient(g_loss, generator.trainable_variables), generator.trainable_variables))
    d_opt.apply_gradients(zip(d_tape.gradient(d_loss, discriminator.trainable_variables), discriminator.trainable_variables))
    return g_loss, d_loss


g_losses, d_losses = [], []
for epoch in range(15):
    for real in data:
        g_loss, d_loss = train_step(real)
    g_losses.append(float(g_loss))
    d_losses.append(float(d_loss))
    print("Epoch", epoch + 1, "- generator loss:", round(g_losses[-1], 3), "discriminator loss:", round(d_losses[-1], 3))

images = generator(tf.random.normal((16, 100))).numpy().reshape(-1, 28, 28) * 0.5 + 0.5
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.axis("off")
plt.suptitle("Generated digits")
plt.show()

plt.plot(g_losses, label="Generator")
plt.plot(d_losses, label="Discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
