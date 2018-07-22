#! /usr/bin/env python3
# coding : utf-8

import tensorflow as tf
import numpy as np
from Data import X_test, y_test, mnist

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

conv1 = tf.layers.conv2d(x, 32, (5, 5), padding="same", activation=tf.nn.relu) 
# 28 x 28 x 32
pool1 = tf.layers.max_pooling2d(conv1, (2, 2), strides=(2, 2))
# 14 x 14 x 32

conv2 = tf.layers.conv2d(pool1, 64, (5, 5), padding="same", activation=tf.nn.relu)
# 14 x 14 x 64
pool2 = tf.layers.max_pooling2d(conv2, (2, 2), strides=(2, 2))
# 7 x 7 x 64

pool_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(pool_flat, 512, activation=tf.nn.relu)
dropout = tf.layers.dropout(dense, rate=0.35)

out = tf.layers.dense(dropout, 10)
yPre = tf.nn.softmax(out)

#loss = tf.losses.sparse_softmax_cross_entropy(y, yPre)
loss = -tf.reduce_sum(y * tf.log(yPre))
train = tf.train.AdamOptimizer().minimize(loss)


accurate = tf.equal(tf.argmax(yPre, 1), tf.argmax(y, 1))
result = tf.reduce_mean(tf.cast(accurate, tf.float32)) * 100

conf = tf.ConfigProto(log_device_placement=True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=conf) as session:
    session.run(tf.global_variables_initializer())
    for i in range(1000):
        X_batch, y_batch = mnist.train.next_batch(100)
        session.run(train, feed_dict={x: X_batch.reshape(100, 28, 28, 1), y: y_batch})
        if i % 50 == 0:
            print(session.run(result, feed_dict={x: X_test, y: y_test}))


