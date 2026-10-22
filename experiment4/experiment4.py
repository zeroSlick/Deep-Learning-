import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

if os.path.isdir("cats_and_dogs_filtered/train"):
    base = "cats_and_dogs_filtered"
    train = keras.utils.image_dataset_from_directory(base + "/train", image_size=(96, 96), batch_size=32, seed=1).take(30)
    test = keras.utils.image_dataset_from_directory(base + "/validation", image_size=(96, 96), batch_size=32, seed=1).take(10).cache()
    print("Dataset: real Cats vs Dogs")
else:
    print("FALLBACK: folder cats_and_dogs_filtered/ not found, using CIFAR-10 cat (0) vs dog (1) images")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    def cat_dog(x, y, n):
        keep = np.isin(y.ravel(), [3, 5])
        x = tf.image.resize(x[keep][:n], (96, 96))
        return tf.data.Dataset.from_tensor_slices((x, (y[keep][:n] == 5).astype("float32"))).batch(32)

    train = cat_dog(x_train, y_train, 1000)
    test = cat_dog(x_test, y_test, 300)

base_model = keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights="imagenet")
base_model.trainable = False

model = keras.Sequential([
    keras.layers.Input((96, 96, 3)),
    keras.layers.Rescaling(1 / 127.5, offset=-1),
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train, epochs=3, verbose=2)
print("Feature extraction test accuracy:", round(model.evaluate(test, verbose=0)[1], 4))

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train, epochs=2, verbose=2)
print("Fine-tuning test accuracy:", round(model.evaluate(test, verbose=0)[1], 4))
