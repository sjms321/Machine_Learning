import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
x_data = digits.data
y_data = digits.target
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
x_train=x_train[200:]
x_valid=x_train[:200]
y_train=y_train[200:]
y_valid=y_train[:200]
class_names = ["0","1", "2", "3", "4", "5","6", "7", "8", "9"]
n_rows = 1
n_cols = 10

plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(x_test[index].reshape(8,8), cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_test[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[8, 8]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
            optimizer="sgd",
            metrics=["accuracy"])
import time
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
start = time.time()
history = model.fit(x_train, y_train, epochs=30,
validation_data=(x_valid, y_valid),callbacks=[tb_hist])
print("time :", time.time() - start)