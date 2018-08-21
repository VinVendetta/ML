#!/usr/bin/env python3
# coding : utf-8

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from getData import data

data = data.values.astype(np.float)

mms = MinMaxScaler()
X = mms.fit_transform(data[:, :-1])
y = data[:, -1].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

score = []
gamma = np.arange(1, 10, 1)
for i in gamma:
    clf = SVC(C=i, gamma=4) # C = 4 gamma=4
    # clf = GradientBoostingClassifier(n_estimators=i) # n_estimators = 140
    clf.fit(X_train, y_train)
    score.append((clf.score(X_train, y_train), clf.score(X_test, y_test)))

plt.plot(gamma, [i for i, j in score], "ro-", label="train")
plt.plot(gamma, [j for i, j in score], "go-", label="test")
for i, s in zip(gamma, score):
    plt.text(i, s[0] + 0.0001, "%.4f" % s[0])
    plt.text(i, s[1] + 0.0001, "%.4f" % s[1])


plt.xlabel("C")
plt.ylabel("Error")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# print("train : ", clf.score(X_train, y_train))
# print("test  : ", clf.score(X_test, y_test))


