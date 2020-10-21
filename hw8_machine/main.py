import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def make_toyimg():
    image = mpimg.imread('edge_detection_ex.jpg')
    image.astype(np.float)
  # print(image.shape)
  # plt.imshow(image.reshape((720,1280,3)), cmap='Greys')
  # plt.show()
    # data format should be change to  batch_shape + [height, width, channels].
    image = image.reshape(1,720, 1280,3)
    image = tf.constant(image, dtype=tf.float64)

    return image

def make_toyfilter():
    weight = np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1],
                       [-1, -2, -1],[0, 0, 0],[1, 2, 1],
                       [-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    weight=weight.reshape((3,3,3,1))
  #  print(weight.shape)
  #  print("weight.shape", weight.shape)
    weight_init = tf.constant_initializer(weight)

    return weight_init

def cnn_valid(image, weight, option,a,b):
    conv2d = keras.layers.Conv2D(filters=1, kernel_size=3, padding=option,kernel_initializer=weight)(image)
 #   print("conv2d.shape", conv2d.shape)
  #  print(conv2d.numpy().reshape(sizeshape,sizeshape1,3))
    plt.imshow(conv2d.numpy().reshape(a,b), cmap='gray')
    plt.show()

def main():
    img =make_toyimg()
    filter = make_toyfilter()
    cnn_valid(img, filter, 'SAME',720,1280)
main()