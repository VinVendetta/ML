#! /usr/bin/env python3
# coding : utf-8



from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn import svm

import numpy as np

iris = datasets.load_iris()

X = iris.data[:,[2,3]]
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
print(ppn.score(X_test_std, y_test))

svc = svm.SVC(decision_function_shape='ovr')
svc.fit(X_train, y_train)
print(svc.score(X_test, y_test))
