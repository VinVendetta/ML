#!/usr/bin/env python3
# coding : utf-8

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

def preProcessY(y):
    yy = np.zeros((y.shape[0], 10), dtype=np.float64)
    for i, value in enumerate(y):
        yy[i, value] = 1.
    return yy

d = load_digits()
X_train, X_test, y_train, y_test = train_test_split(d.data, d.target, test_size=0.3)


y_train = preProcessY(y_train)
y_test = preProcessY(y_test)
#print(y_train)
#print(y_test)
# X_train = X_train.astype(np.float32)
# X_test = X_test.astype(np.float32)