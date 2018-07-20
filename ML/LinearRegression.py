#! /usr/bin/env python3
# coding : utf-8

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split


clf = LinearRegression() 
data = datasets.load_boston()
X = data.data
y = data.target
# X_train = X[:int(len(X) * 0.7)]
# X_test = X[int(len(X) * 0.7):]
# y_train = y[:int(len(y) * 0.7)]
# y_test = y[int(len(y) * 0.7):]
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

clf.fit(X_train, y_train)
print(clf.coef_, clf.intercept_)
print(clf.score(X_test, y_test))





# from sklearn import datasets
# from sklearn.model_selection import cross_val_predict
# from sklearn import linear_model
# import matplotlib.pyplot as plt

# lr = linear_model.LinearRegression()
# boston = datasets.load_boston()
# y = boston.target

# # cross_val_predict returns an array of the same size as `y` where each entry
# # is a prediction obtained by cross validation:
# predicted = cross_val_predict(lr, boston.data, y, cv=10)

# fig, ax = plt.subplots()
# ax.scatter(y, predicted, edgecolors=(0, 0, 0))
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()

