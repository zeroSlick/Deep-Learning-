import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train[:8000] / 255.0
y_train = y_train[:8000]
x_test = x_test[:2000] / 255.0
y_test = y_test[:2000]


def run(init, optimizer):
    model = keras.Sequential([
        keras.layers.Input((32, 32, 3)),
        keras.layers.Conv2D(32, 3, activation="relu", kernel_initializer=init),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu", kernel_initializer=init),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu", kernel_initializer=init),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)
    return model.evaluate(x_test, y_test, verbose=0)


print("Initialization comparison (Adam)")
for init in ["glorot_uniform", "he_normal"]:
    loss, acc = run(init, "adam")
    print(init, "- loss:", round(loss, 4), "accuracy:", round(acc, 4))

print("Optimizer comparison (He init)")
for name in ["sgd", "adam", "rmsprop"]:
    loss, acc = run("he_normal", name)
    print(name, "- loss:", round(loss, 4), "accuracy:", round(acc, 4))

print("Learning-rate schedule (Adam with exponential decay)")
schedule = keras.optimizers.schedules.ExponentialDecay(0.005, decay_steps=200, decay_rate=0.9)
loss, acc = run("he_normal", keras.optimizers.Adam(schedule))
print("schedule - loss:", round(loss, 4), "accuracy:", round(acc, 4))
