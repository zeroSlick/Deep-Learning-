import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)

x, y = load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = keras.Sequential([
    keras.layers.Input((13,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, keras.utils.to_categorical(y_train), epochs=50, batch_size=16, verbose=0)

pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred, average="macro"), 4))
print("Recall:", round(recall_score(y_test, pred, average="macro"), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
