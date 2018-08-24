#!/usr/bin/env python3
# coding : utf-8

from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


from getData import data
import matplotlib.pyplot as plt
import numpy as np

# ss = StandardScaler(copy=False)
# ss.fit_transform(data[3:, :-1])

pca = PCA(copy=False)
pca.fit_transform(data[3:, :-1])

X_train1, X_test1, y_train1, y_test1 = train_test_split(data[:, :-1], data[:, -1], test_size=0.3)
X_train2, X_test2, y_train2, y_test2 = train_test_split(data[:, :5], data[:, -1], test_size=0.3)
X_train3, X_test3, y_train3, y_test3 = train_test_split(data[:, 6:-1], data[:, -1], test_size=0.3)

d = [(X_train1, y_train1, X_test1, y_test1), 
	 (X_train2, y_train2, X_test2, y_test2), 
	 (X_train3, y_train3, X_test3, y_test3)]

def getScore(clf, X_test, y_test):
	yPre = np.round(clf.predict(X_test))
	# print(np.hstack((yPre, y_test.reshape(X_test.shape[0], 1))))
	acc = (yPre == y_test)
	print(f"{np.mean(acc) * 100:*^20}")


for clf in [SVC(), GradientBoostingClassifier(n_estimators=3, max_depth=2, learning_rate=0.5)]:
	X_train, y_train, X_test, y_test = d[0]
	clf.fit(X_train, y_train)
	print(f"{'test': ^10}{accuracy_score(y_test, clf.predict(X_test)) * 100}") 
	#getScore(clf, X_test, y_test)
	print(f"{'train': ^10}{accuracy_score(y_train, clf.predict(X_train)) * 100}")
	#getScore(clf, X_train, y_train)
	print("*" * 30)
	print(cross_val_score(clf, X_test, y_test))

# for clf in [RandomForestRegressor(n_estimators=20, max_depth=5), SVR(),
#     DecisionTreeRegressor(max_depth=5), LinearRegression()]:

#     clf.fit(X_train1, y_train1)
#     getScore(clf, X_test1, y_test1)
#     print()







# labels = ["Sex", "Length", "Diamter", "Height", "WholeWeight", 
#         "ShuckedWeight", "VisceraWeight", "SellWeight", "Rings"]
# for i in range(8):
#     for j in range(i + 1, 8):
#         plt.scatter(data[:, i], data[:, j + 1], c=np.random.rand(4177), linewidths=0.0001)
#         plt.xlabel(labels[i])
#         plt.ylabel(labels[j + 1])
#         plt.show()
# for i in range(8):
#         plt.scatter(data[:, i], data[:, -1], c=np.random.rand(4177), linewidths=0.0001)
#         plt.xlabel(labels[i])
#         plt.ylabel("Rings")
#         plt.show()
	

