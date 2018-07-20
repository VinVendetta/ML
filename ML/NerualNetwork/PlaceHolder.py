#! /usr/bin/env python3
# coding : utf-8

import tensorflow as tf
import numpy as np

p1 = tf.placeholder(tf.float32)
p2 = tf.placeholder(tf.float32)

with tf.Session() as session:
    print(session.run(tf.add(p1, p2), feed_dict={p1: 3.0, p2: 4.0}))



