#! /usr/bin/env python3
# coding : utf-8

import tensorflow as tf
import numpy as np

matrix1 = np.ones((1,2)) * 3
matrix2 = np.ones((2,1)) * 2

matrix1 = tf.constant(matrix1)
matrix2 = tf.constant(matrix2)


with tf.Session() as session:
    print(session.run(tf.matmul(matrix1, matrix2)))
    
# print(np.dot(matrix1, matrix2))



