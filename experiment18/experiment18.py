import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

if os.path.isdir("cats_and_dogs_filtered/train"):
    base = "cats_and_dogs_filtered"
    train = keras.utils.image_dataset_from_directory(base + "/train", image_size=(64, 64), batch_size=32, seed=1).take(40)
    test = keras.utils.image_dataset_from_directory(base + "/validation", image_size=(64, 64), batch_size=32, seed=1).take(20).cache()
    print("Dataset: real Cats vs Dogs")
else:
    print("FALLBACK: folder cats_and_dogs_filtered/ not found, using CIFAR-10 cat (0) vs dog (1) images")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    def cat_dog(x, y, n):
        keep = np.isin(y.ravel(), [3, 5])
        return tf.data.Dataset.from_tensor_slices((x[keep][:n].astype("float32"), (y[keep][:n] == 5).astype("float32"))).batch(32)

    train = cat_dog(x_train, y_train, 3000)
    test = cat_dog(x_test, y_test, 1000)

shape = next(iter(train))[0].shape[1:]

for name, augment in [("Without augmentation", False), ("With augmentation", True)]:
    model = keras.Sequential([keras.layers.Input(shape)])
    if augment:
        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.1))
        model.add(keras.layers.RandomZoom(0.1))
    model.add(keras.layers.Rescaling(1 / 255))
    model.add(keras.layers.Conv2D(32, 3, activation="relu"))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Conv2D(64, 3, activation="relu"))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train, epochs=8, verbose=0)
    print(name, "- test accuracy:", round(model.evaluate(test, verbose=0)[1], 4))
