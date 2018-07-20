#! /usr/bin/env python3
# coding : utf-8

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from Data import X_test, X_train, y_test, y_train
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float64, [None, 64], name="x")
y = tf.placeholder(tf.float64, [None, 10], name="y")

W = tf.Variable(tf.random_normal([64, 10], dtype=tf.float64))
b = tf.Variable(tf.zeros([10], dtype=tf.float64) + 0.1)

yPre = tf.nn.softmax(tf.matmul(x, W) + b)

loss = -tf.reduce_sum(y * tf.log(yPre))
#train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


# accPercent = []
# for i in np.linspace(0.01, 0.05, num=20):
#     train = tf.train.AdamOptimizer(learning_rate=i).minimize(loss)
#     init = tf.global_variables_initializer()
#     with tf.Session() as session:
#         session.run(init)
#         for i in range(1000):
#             session.run(train, feed_dict={x: X_train, y: y_train})

#         pred = session.run(yPre, feed_dict={x: X_test})
#         accurate = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
#         result = tf.reduce_mean(tf.cast(accurate, tf.float32))
#         accPercent.append(100 * session.run(result, feed_dict={x: X_test, y: y_test}))

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(np.linspace(0.01, 0.05, num=20), accPercent, lw='2') # learning_rate = 0.045(95%)
# plt.show()

train = tf.train.AdamOptimizer(learning_rate=0.045).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(1000):
        session.run(train, feed_dict={x: X_train, y: y_train})

    pred = session.run(yPre, feed_dict={x: X_test})
    accurate = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    result = tf.reduce_mean(tf.cast(accurate, tf.float32))
    print(f'result = {100 * session.run(result, feed_dict={x: X_test, y: y_test})} %')

