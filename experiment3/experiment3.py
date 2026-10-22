from tensorflow import keras
import tensorflow as tf

tf.random.set_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train[:10000] / 255.0
y_train = y_train[:10000]
x_test = x_test[:2000] / 255.0
y_test = y_test[:2000]

model = keras.Sequential([
    keras.layers.Input((32, 32, 3)),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", round(acc, 4))

model.save("cifar_cnn.keras")
loaded = keras.models.load_model("cifar_cnn.keras")
loss, acc = loaded.evaluate(x_test, y_test, verbose=0)
print("Reloaded model test accuracy:", round(acc, 4))
