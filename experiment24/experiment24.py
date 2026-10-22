import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
tf.random.set_seed(0)

if os.path.exists("creditcard.csv"):
    print("Dataset: real Credit Card Fraud (creditcard.csv)")
    df = pd.read_csv("creditcard.csv")
    x, y = df.drop(columns="Class").values, df["Class"].values
else:
    print("FALLBACK: creditcard.csv not found. Using synthetic transactions (normal = structured, anomaly = random)")
    normal = rng.normal(size=(10000, 3)) @ rng.normal(size=(3, 10)) + rng.normal(0, 0.1, (10000, 10))
    anomaly = rng.normal(0, 2, (200, 10))
    x = np.vstack([normal, anomaly])
    y = np.array([0] * 10000 + [1] * 200)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
scaler = StandardScaler().fit(x_train[y_train == 0])
x_train = scaler.transform(x_train[y_train == 0])
x_test = scaler.transform(x_test)
n = x_train.shape[1]

model = keras.Sequential([
    keras.layers.Input((n,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(3, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(n),
])
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, x_train, epochs=20, batch_size=128, verbose=0)

train_error = np.mean((x_train - model.predict(x_train, verbose=0)) ** 2, axis=1)
test_error = np.mean((x_test - model.predict(x_test, verbose=0)) ** 2, axis=1)
threshold = np.percentile(train_error, 99)
pred = (test_error > threshold).astype(int)

print("Threshold (99th percentile of normal training error):", round(threshold, 4))
print("Precision:", round(precision_score(y_test, pred), 4))
print("Recall:", round(recall_score(y_test, pred), 4))
print("F1-score:", round(f1_score(y_test, pred), 4))
