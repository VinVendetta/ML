#!/usr/bin/env python3
# coding : utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("./abalone_data.txt", names=["Sex", "Length", "Diamter", "Height", 
    "WholeWeight", "ShuckedWeight", "VisceraWeight", "SellWeight", "Rings"])

# data.replace(to_replace="M", value="1", inplace=True)
# data.replace(to_replace="F", value="0", inplace=True)
# data.replace(to_replace="I", value="0.5", inplace=True)

# array = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 29]
def fun(x):
    array = np.around(np.geomspace(1, 29, 5))
    for i in range(len(array)):
        if x <= array[0]:
            return 0
        if array[i] <= x <= array[i + 1]:
            return i + 1
        if x >= array[-1]:
            return len(array)

# print(data.Rings)
data["Rings"] = data["Rings"].map(fun) #[i for i in map(fun, data["Rings"])
# print(data.Rings)

Sex = data.pop("Sex")
I = Sex.copy()
I.replace(to_replace="I", value="1", inplace=True)
I.replace(to_replace="M", value="0", inplace=True)
I.replace(to_replace="F", value="0", inplace=True)

M = Sex.copy()
M.replace(to_replace="I", value="0", inplace=True)
M.replace(to_replace="M", value="1", inplace=True)
M.replace(to_replace="F", value="0", inplace=True)

F = Sex.copy()
F.replace(to_replace="I", value="0", inplace=True)
F.replace(to_replace="M", value="0", inplace=True)
F.replace(to_replace="F", value="1", inplace=True)

data.insert(0, "I", I)
data.insert(0, "F", F)
data.insert(0, "M", M)
# print(data.columns)


# counts = []
print(data)
for i in range(1, 30):
    count = data.loc[lambda df: df["Rings"] == float(i)]
    print(i, count.shape[0])


# data = data.values.astype(np.float)

