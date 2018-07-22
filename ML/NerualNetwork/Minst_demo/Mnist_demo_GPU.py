#! /usr/bin/env python3
# coding : utf-8

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from Data import X_test, X_train, y_test, y_train
import numpy as np
import matplotlib.pyplot as plt
import os
import time

startTime = time.time()

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

x = tf.placeholder(tf.float64, [None, 64], name="x")
y = tf.placeholder(tf.float64, [None, 10], name="y")

W = tf.Variable(tf.random_normal([64, 10], dtype=tf.float64))
b = tf.Variable(tf.zeros([10], dtype=tf.float64) + 0.1)

yPre = tf.nn.softmax(tf.matmul(x, W) + b)

loss = -tf.reduce_sum(y * tf.log(yPre))


train = tf.train.AdamOptimizer(learning_rate=0.045).minimize(loss)
init = tf.global_variables_initializer()

conf = tf.ConfigProto(log_device_placement=True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=conf) as session:
    session.run(init)
    for i in range(1000):
        session.run(train, feed_dict={x: X_train, y: y_train})

    pred = session.run(yPre, feed_dict={x: X_test})
    accurate = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    result = tf.reduce_mean(tf.cast(accurate, tf.float32))
    print(f'result = {100 * session.run(result, feed_dict={x: X_test, y: y_test})} %')

print("used Time ", time.time() - startTime)