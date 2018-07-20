#!/usr/bin/env python3
# coding : utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def addLayer(inputs, inputSize, outputSize, activeFunction=None):
    W = tf.Variable(np.random.randn(inputSize, outputSize).astype(np.float32))
    b = tf.Variable(np.zeros((1, outputSize)).astype(np.float32))
    yPre = tf.matmul(inputs, W) + b
    return activeFunction(yPre) if activeFunction is not None else yPre


Xs = tf.placeholder(np.float32, [None, 1])
Ys = tf.placeholder(np.float32, [None, 1])

X_data = np.linspace(-1, 1, 300, dtype=np.float32).reshape(300, 1)
# noise = np.random.randn(300, 1).astype(np.float32)
noise = np.random.normal(0, 0.05, size=X_data.shape)
y_data = np.square(X_data) + 0.7 + noise

l1 = addLayer(Xs, 1, 10, tf.nn.relu)
yPre = addLayer(l1, 10, 1)

loss =  tf.reduce_mean(tf.reduce_sum(tf.square(Ys - yPre), axis=[1]))
train = tf.train.AdamOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X_data, y_data)
plt.ion()
plt.show()

with tf.Session() as session:
    session.run(init)
    for _ in range(1000):
        session.run(train, feed_dict={Xs: X_data, Ys: y_data})
        if _ % 40 == 0:
            #print(session.run(loss, feed_dict={Xs: X_data, Ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            lines = ax.plot(X_data, session.run(yPre, feed_dict={Xs: X_data}), 'r-', lw='2')
            
            plt.pause(0.2)


