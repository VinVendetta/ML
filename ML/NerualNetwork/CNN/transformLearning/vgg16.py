#!/usr/bin/env python3
# coding : utf-8

import numpy as np
import tensorflow as tf
from getData import X_train, X_test, y_train, y_test


class Vgg:
    def __init__(self, parmPath="../vgg16.npy"):
        self.X = tf.placeholder(tf.float32, [None, 224, 224, 3], name="X")
        self.y = tf.placeholder(tf.float32, [None, 3], name="y")
        self.preTrainedParm = np.load(parmPath, encoding="latin1").item() 
        self.build()

    def getBias(self, name):
        return self.preTrainedParm[name][1]
    
    def getWeightOrFilter(self, name):
        return self.preTrainedParm[name][0]
    
    def conv2dLayer(self, input, name):
        return tf.nn.relu(tf.nn.conv2d(input, self.getWeightOrFilter(name), [1, 1, 1, 1], padding="SAME") + self.getBias(name))

    def maxPoolLayer(self, input):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    
    def fullConnectedLayer(self, input, name):
        W = self.getWeightOrFilter(name)
        b = self.getBias(name)
        dim = 1
        for i in input.shape[1:]:
            dim *= i

        return tf.matmul(tf.reshape(input, [-1, dim]), W) + b

    def build(self):
        self.conv1_1 = self.conv2dLayer(self.X, "conv1_1")
        self.conv1_2 = self.conv2dLayer(self.conv1_1, "conv1_2")
        self.maxpool1 = self.maxPoolLayer(self.conv1_2) # 112 * 112 * 64 

        self.conv2_1 = self.conv2dLayer(self.maxpool1, "conv2_1")
        self.conv2_2 = self.conv2dLayer(self.conv2_1, "conv2_2")
        self.maxpool2 = self.maxPoolLayer(self.conv2_2) # 56 * 56 * 128

        self.conv3_1 = self.conv2dLayer(self.maxpool2, "conv3_1")
        self.conv3_2 = self.conv2dLayer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv2dLayer(self.conv3_2, "conv3_3")
        self.maxpool3 = self.maxPoolLayer(self.conv3_3) # 28 * 28 * 256

        self.conv4_1 = self.conv2dLayer(self.maxpool3, "conv4_1")
        self.conv4_2 = self.conv2dLayer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv2dLayer(self.conv4_2, "conv4_3")
        self.maxpool4 = self.maxPoolLayer(self.conv4_3) # 14 * 14 * 512

        self.conv5_1 = self.conv2dLayer(self.maxpool4, "conv5_1")
        self.conv5_2 = self.conv2dLayer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv2dLayer(self.conv5_2, "conv5_3")
        self.maxpool5 = self.maxPoolLayer(self.conv5_3) # 7 * 7 * 512

        self.fc6 = tf.nn.relu(self.fullConnectedLayer(self.maxpool5, "fc6"))
        self.fc7 = tf.nn.relu(self.fullConnectedLayer(self.fc6, "fc7"))
        self.fc8 = tf.layers.dense(self.fc7, 3, activation=tf.nn.relu)
        
    def train(self):
        yPre = tf.nn.softmax(self.fc8)
        loss = tf.losses.softmax_cross_entropy(self.y, yPre)
        train = tf.train.AdamOptimizer().minimize(loss)

        accurate = tf.equal(tf.argmax(yPre, 1), tf.argmax(self.y, 1))
        result = tf.reduce_mean(tf.cast(accurate, tf.float32)) * 100
        # images, labels = getImages("/Users/apple/Desktop/pic/", count=-1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                sess.run(train, feed_dict={self.X: X_train, self.y: y_train})
                if i % 50 == 0:
                    acc = sess.run(result, {self.X: X_test, self.y: y_test})
                    print(acc)
            saver.save(sess, "./carPersonBike.npy") 


vgg = Vgg()
vgg.train()
# print(getImages("/Users/apple/Desktop/pic/").shape)


