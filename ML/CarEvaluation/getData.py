#!/usr/bin/env python3
# coding : utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("./data.txt")

# | names file (C4.5 format) for car evaluation domain

# | class values

# unacc, acc, good, vgood

# | attributes

# buying:   vhigh, high, med, low.
# maint:    vhigh, high, med, low.
# doors:    2, 3, 4, 5more.
# persons:  2, 4, more.
# lug_boot: small, med, big.
# safety:   low, med, high.
#
# 9. Class Distribution (number of instances per class)

#    class      N          N[%]
#    -----------------------------
#    unacc     1210     (70.023 %) 
#    acc        384     (22.222 %) 
#    good        69     ( 3.993 %) 
#    v-good      65     ( 3.762 %) 
 

data.buying.replace(to_replace={"vhigh": 4, "high": 3, "med": 2, "low": 1}, inplace=True)
data.maint.replace(to_replace={"vhigh": 4, "high": 3, "med": 2, "low": 1}, inplace=True)
data.doors.replace(to_replace={"5more": 5}, inplace=True)
data.persons.replace(to_replace={"more": 6}, inplace=True)
data.lug_boot.replace(to_replace={"small": 1, "med": 2, "big": 3}, inplace=True)
data.safety.replace(to_replace={"low": 1, "med": 2, "high": 3}, inplace=True)
data.evaluation.replace(to_replace={"unacc": 1, "acc": 2, "good": 3, "vgood": 4}, inplace=True)

# evaluationMeans = [data.loc[lambda df: df.buying == int("%d" % i), lambda df: df.columns[-1]].mean(axis=1) for i in range(1, 5)]
# corr = data.corr()
# print(corr, corr.shape, data.shape, sep="\n")

# print(data.shape, data.columns, data.index, sep='\n')
# print(data[:10, :], data.dtype)



