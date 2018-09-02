#!/usr/bin/env python3
# coding : utf-8

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split


def getImages(dirPath, count=10):
    fileNames = list(map(lambda x: dirPath + x, filter(lambda x:".jpg" in x or ".png" in x or ".jpeg" in x, os.listdir(dirPath))))
    images = []
    if count != -1:
        fileNames = fileNames[:count]
    labels = np.zeros((len(fileNames), 3), dtype=np.int) # car, person, bike
    for i in range(len(fileNames)):
        image = cv2.imread(fileNames[i])
        image = cv2.resize(image, (224, 224))
        images.append(image)

        if "car" in fileNames[i]:
            labels[i, 0] = 1
        if "person" in fileNames[i]:
            labels[i, 1] = 1
        if "bike" in fileNames[i]:
            labels[i, 2] = 1

    return np.array(images), labels

images, labels = getImages("/Users/apple/Desktop/pic/", count=-1)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
