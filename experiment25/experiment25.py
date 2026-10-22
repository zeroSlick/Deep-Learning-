import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

np.random.seed(0)
tf.random.set_seed(0)
SIZE = 64

try:
    import tensorflow_datasets as tfds
    data = tfds.load("oxford_iiit_pet", split="train[:400]")
    images, masks = [], []
    for item in data:
        images.append(tf.image.resize(item["image"], (SIZE, SIZE)).numpy() / 255.0)
        masks.append((tf.image.resize(item["segmentation_mask"], (SIZE, SIZE), method="nearest").numpy() == 1).astype("float32"))
    images, masks = np.array(images), np.array(masks)
    print("Dataset: real Oxford-IIIT Pet (400 images, mask = pet vs background)")
except Exception:
    print("FALLBACK: Oxford-IIIT Pet unavailable (pip install tensorflow-datasets, ~800MB download). Using synthetic shape images.")
    images = np.random.rand(400, SIZE, SIZE, 3).astype("float32") * 0.3
    masks = np.zeros((400, SIZE, SIZE, 1), dtype="float32")
    yy, xx = np.mgrid[:SIZE, :SIZE]
    for i in range(400):
        cy, cx, r = np.random.randint(16, 48, 2).tolist() + [np.random.randint(8, 14)]
        shape = ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2) if i % 2 == 0 else ((abs(yy - cy) < r) & (abs(xx - cx) < r))
        masks[i, :, :, 0] = shape
        images[i][shape] = np.random.rand(3) * 0.5 + 0.5

x_train, y_train, x_test, y_test = images[:320], masks[:320], images[320:], masks[320:]

inputs = keras.Input((SIZE, SIZE, 3))
c1 = keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
p1 = keras.layers.MaxPooling2D()(c1)
c2 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
p2 = keras.layers.MaxPooling2D()(c2)
c3 = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
u2 = keras.layers.Concatenate()([keras.layers.UpSampling2D()(c3), c2])
c4 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(u2)
u1 = keras.layers.Concatenate()([keras.layers.UpSampling2D()(c4), c1])
c5 = keras.layers.Conv2D(16, 3, activation="relu", padding="same")(u1)
outputs = keras.layers.Conv2D(1, 1, activation="sigmoid")(c5)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=2)

pred = (model.predict(x_test, verbose=0) > 0.5).astype("float32")
intersection = np.sum(pred * y_test)
union = np.sum(pred) + np.sum(y_test) - intersection
print("IoU:", round(float(intersection / union), 4))

plt.figure(figsize=(8, 8))
for i in range(3):
    for col, (title, img) in enumerate([("Image", x_test[i]), ("True mask", y_test[i, :, :, 0]), ("Predicted mask", pred[i, :, :, 0])]):
        plt.subplot(3, 3, i * 3 + col + 1)
        plt.imshow(img, cmap=None if col == 0 else "gray")
        plt.title(title)
        plt.axis("off")
plt.show()
