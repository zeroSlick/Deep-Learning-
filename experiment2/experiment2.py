import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)

x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


def build(reg=None, dropout=0.0):
    model = keras.Sequential([
        keras.layers.Input((8,)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss="mse", metrics=["mae"])
    return model


models = {
    "Baseline": build(),
    "L1": build(keras.regularizers.l1(0.001)),
    "L2": build(keras.regularizers.l2(0.001)),
    "Dropout": build(dropout=0.2),
}

for name, model in models.items():
    model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=0)
    loss, mae = model.evaluate(x_test, y_test, verbose=0)
    print(name, "- test MSE:", round(loss, 4), "test MAE:", round(mae, 4))
