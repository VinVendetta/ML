#! /usr/bin/env python3
# coding : utf-8

import tensorflow as tf
import numpy as np

var1 = tf.Variable(np.arange(9).reshape(3,3).astype(np.float32))
var2 = tf.constant(np.ones((3,3), dtype=np.float32).T * 10.)

newVar = tf.matmul(var1, var2)
update = tf.assign(var1, newVar)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for _ in range(3):
        session.run(update)
        print(session.run(var1))

