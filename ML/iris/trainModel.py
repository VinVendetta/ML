#! /usr/bin/env python3
# coding : utf-8

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt 
from sklearn.externals import joblib

from dataPreprocessing import getData


X, y = getData()
#X = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

#clf = LinearRegression()
clf = SVC(kernel='rbf', decision_function_shape='ovo')
#clf = LinearSVC(dual=False)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
#joblib.dump(clf, "./Iris.model")
score = []
for i in range(3, 20):
    score.append(cross_val_score(clf, X, y, cv=i).mean())

plt.title('Cross Validation')
plt.plot(range(3,20), score)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()

