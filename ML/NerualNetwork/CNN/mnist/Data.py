#! /usr/bin/env python3
# coding : utf-8

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# X_train = mnist.train.images.reshape(55000, 28, 28, 1) # shape (55000, 784)
X_test = mnist.test.images.reshape(10000, 28, 28, 1)   # shape (10000, 784)
# y_train = mnist.train.images                           # shape (55000, 10)
y_test = mnist.test.labels                             # shape (10000, 10)
