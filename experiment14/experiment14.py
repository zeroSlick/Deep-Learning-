import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)

x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

for activation in ["relu", "tanh"]:
    for optimizer in ["adam", "sgd"]:
        model = keras.Sequential([
            keras.layers.Input((30,)),
            keras.layers.Dense(16, activation=activation),
            keras.layers.Dense(8, activation=activation),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=30, batch_size=16, verbose=0)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(activation, "+", optimizer, "- test loss:", round(loss, 4), "test accuracy:", round(acc, 4))
