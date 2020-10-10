'''
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

'''
import sys

assert sys.version_info >= (3, 5)
from tensorflow import keras
# 사이킷런 ≥0.20 필수
import sklearn

assert sklearn.__version__ >= "0.20"

# 텐서플로 ≥2.0 필수
import tensorflow as tf

assert tf.__version__ >= "2.0"

# 공통 모듈 임포트
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time
import pandas as pd



# 클래스 이름은 다음과 같습니다:

class_names = ["0","1", "2", "3", "4", "5","6", "7", "8", "9"]


def load_data():
    # 먼저 MNIST 데이터셋을 로드하겠습니다. 케라스는 `keras.datasets`에 널리 사용하는 데이터셋을 로드하기 위한 함수를 제공합니다. 이 데이터셋은 이미 훈련 세트와 테스트 세트로 나누어져 있습니다. 훈련 세트를 더 나누어 검증 세트를 만드는 것이 좋습니다:
    digits = load_digits()
    X_data = digits.data
    y_data = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    X_train = X_train[200:]
    X_valid = X_train[:200]
    y_train = y_train[200:]
    y_valid = y_train[:200]

    return  X_train,y_train,X_valid, y_valid, X_test, y_test


def show_10images(X_train, y_train):
    # 이 데이터셋에 있는 샘플 이미지를 몇 개 출력해 보죠:
    n_rows = 1
    n_cols = 10
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(X_train[index].reshape(8,8), cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[y_train[index]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()

def dense(label_dim,weight_init,activation):
    return tf.keras.layers.Dense(units=label_dim, use_bias=True, kernel_initializer=weight_init,activation=activation)


def makemodel1(X_train, y_train, X_valid, y_valid):
    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=[8, 8]))

    model.add(keras.layers.Dense(300, activation="relu"))

    model.add(keras.layers.Dense(100, activation="relu"))

    model.add(keras.layers.Dense(10, activation="softmax"))

    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    return model
def makemodel(X_train, y_train, X_valid, y_valid, weight_init):
    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=[8, 8]))

    model.add(dense(300,weight_init, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(dense(100,weight_init, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(dense(10,weight_init, activation="softmax"))

    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    return model

def modelprdict(model,X_train, y_train, X_test, y_test):
    tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    start = time.time()
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_test, y_test), callbacks=[tb_hist])
    print("time :", time.time() - start)
    return history
def evalmodel(model, history, X_test, y_test):
    model.evaluate(X_test, y_test)

    X_new = X_test[:3]
    y_proba = model.predict(X_new)
    y_proba.round(2)

    #y_pred = model.predict_classes(X_new)
    #y_pred

    plt.figure(figsize=(7.2, 2.4))
    for index, image in enumerate(X_new):
        plt.subplot(1, 3, index + 1)
        plt.imshow(image.reshape(8,8), cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_test[index]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    #save_fig("keras_learning_curves_plot")
    #plt.show()
    #plt.show()

def plot_history(histories,key='accuracy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch,history.history['val_'+key],
                       '--',label=name.title()+' val')
        plt.plot(history.epoch,history.history[key],color=val[0].get_color(),label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.show()


def main():
    X_train, y_train, X_valid,y_valid,X_test, y_test = load_data()
    show_10images(X_train, y_train)
  
    model_xavier = makemodel(X_train, y_train, X_valid, y_valid,'glorot_uniform')
    hist_xavier=modelprdict(model_xavier,X_train, y_train, X_test, y_test)
    x=makemodel1(X_train, y_train, X_valid, y_valid)
    hist_Random=modelprdict(model_xavier,X_train, y_train, X_test, y_test)

    plot_history([('Random',hist_Random),('Dropout weight init',hist_xavier)])
    plt.show()


main()
