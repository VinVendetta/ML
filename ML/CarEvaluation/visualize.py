#!/usr/bin/env python3
# coding : utf-8

import matplotlib.pyplot as plt
from getData import data

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

# 相关度 
#
#              buying     maint  lug_boot    safety  evaluation
# buying      1.00000  0.000000  0.000000  0.000000   -0.282750
# maint       0.00000  1.000000  0.000000  0.000000   -0.232422
# lug_boot    0.00000  0.000000  1.000000  0.000000    0.157932
# safety      0.00000  0.000000  0.000000  1.000000    0.439337               
# evaluation -0.28275 -0.232422  0.157932  0.439337    1.0000

buying = data.iloc[:, 0]
maint = data.iloc[:, 1]
lug_boot = data.iloc[:, 4]
safety = data.iloc[:, -2]
evaluation = data.iloc[:, -1]


font = {'family' : 'serif',
        'color'  : 'red',
        'weight' : 'normal',
        'size'   : 12,}

# buying : evaluation
ax = plt.subplot("221")
evaluationMeans = [data.loc[lambda df: df.buying == int("%d" % i), 
    lambda df: df.columns[-1]].mean() for i in range(1, 5)]

plt.bar(range(1, 5), evaluationMeans, 0.35)
plt.xticks(range(1, 5), ["vhigh", "high", "med", "low"][::-1])
plt.xlabel("buying", fontdict=font)
plt.ylabel("Means of evaluation", fontdict=font)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for x, y in zip(range(1, 5), evaluationMeans):
    plt.text(x - 0.25, y + 0.05, "%.2f" % y)

# maint : evaluation 
ax = plt.subplot("222")
evaluationMeans = [data.loc[lambda df: df.maint == int("%d" % i), 
    lambda df: df.columns[-1]].mean() for i in range(1, 5)]

plt.bar(range(1, 5), evaluationMeans, 0.35)
plt.xticks(range(1, 5), ["vhigh", "high", "med", "low"][::-1])
plt.xlabel("maint", fontdict=font)
plt.ylabel("Means of evaluation", fontdict=font)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for x, y in zip(range(1, 5), evaluationMeans):
    plt.text(x - 0.25, y + 0.05, "%.2f" % y)


# lug_boot : evaluation 

ax = plt.subplot("223")
evaluationMeans = [data.loc[lambda df: df.lug_boot == int("%d" % i), 
    lambda df: df.columns[-1]].mean() for i in range(1, 4)]

plt.bar(range(1, 4), evaluationMeans, 0.35)
plt.xticks(range(1, 5), ["high", "med", "low"][::-1])
plt.xlabel("lug_boot", fontdict=font)
plt.ylabel("Means of evaluation", fontdict=font)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for x, y in zip(range(1, 5), evaluationMeans):
    plt.text(x - 0.25, y + 0.05, "%.2f" % y)


# safety : evaluation

ax = plt.subplot("224")
evaluationMeans = [data.loc[lambda df: df.safety == int("%d" % i), lambda df: df.columns[-1]].mean() for i in range(1, 4)]
plt.bar(range(1, 4), evaluationMeans, 0.35)
plt.xticks(range(1, 5), ["high", "med", "low"][::-1])
plt.xlabel("safety", fontdict=font)
plt.ylabel("Means of evaluation", fontdict=font)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for x, y in zip(range(1, 5), evaluationMeans):
    plt.text(x - 0.25, y + 0.05, "%.2f" % y)


plt.tight_layout()
plt.show()
