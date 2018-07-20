#! /usr/bin/env python3
# coding : utf-8

import numpy as np
from sklearn import datasets
import pandas as pd
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE


iris = datasets.load_iris()
X = iris.data
y = iris.target


def getData():
    data = pd.DataFrame(X, 
        columns=['sepal_length', 'separl_width', 'petal_length', 'petal_width'],
        copy=True
    )
    # data.join(pd.Series(y, name='type'), how='right')
    data['type'] = y;
    data = data.values
    np.random.shuffle(data)
    return data[:,:-1], data[:,-1] # X, y


# plt.subplot(221)
# plt.title("sepal_length")
# plt.scatter(X[:,0], y, c='red', )

# plt.subplot(222)
# plt.title("sepal_width")
# plt.scatter(X[:,1], y, c='blue')

# plt.subplot(223)
# plt.title("petal_length")
# plt.scatter(X[:,2], y, c='green')

# plt.subplot(224)
# plt.title("petal_width")
# plt.scatter(X[:,3], y, c='black')
# plt.show()
# plt.scatter([i for i in range(10)], [2 * j for j in range(10)])





