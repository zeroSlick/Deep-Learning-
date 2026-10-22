import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

np.random.seed(0)
tf.random.set_seed(0)
URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

try:
    path = keras.utils.get_file("jena_climate_2009_2016.csv.zip", URL, extract=True)
    csv = os.path.join(path if os.path.isdir(path) else os.path.dirname(path), "jena_climate_2009_2016.csv")
    temp = pd.read_csv(csv)["T (degC)"].values[::6][:5000]
    print("Dataset: real Jena Climate (hourly, first 5000 hours)")
except Exception:
    print("FALLBACK: Jena Climate unavailable, using a synthetic daily-cycle temperature series")
    t = np.arange(5000)
    temp = 10 + 8 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 1, 5000)

split = int(0.8 * len(temp))
mean, std = temp[:split].mean(), temp[:split].std()
scaled = (temp - mean) / std

window = 24
x = np.array([scaled[i:i + window] for i in range(len(scaled) - window)])[..., None]
y = scaled[window:]
x_train, y_train = x[:split - window], y[:split - window]
x_test, y_test = x[split - window:], y[split - window:]

model = keras.Sequential([
    keras.layers.Input((window, 1)),
    keras.layers.LSTM(32),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=2)

pred = model.predict(x_test, verbose=0).ravel() * std + mean
actual = y_test * std + mean
print("MAE:", round(np.mean(np.abs(actual - pred)), 4), "degC")
print("RMSE:", round(np.sqrt(np.mean((actual - pred) ** 2)), 4), "degC")

plt.plot(actual[:200], label="Actual")
plt.plot(pred[:200], label="Predicted")
plt.xlabel("Hour")
plt.ylabel("Temperature (degC)")
plt.legend()
plt.show()
