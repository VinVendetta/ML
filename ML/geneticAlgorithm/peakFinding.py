#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def F(x):
    # return 2.4 * x ** (0.3*x+1.7) - (10 * x - 1.2) * np.sin(0.2 * x ** x)
    return np.sin(3 * x + 7) - np.cos(0.3 * x ** 2 - 1.5)

x = np.linspace(-1, 6, num=500)

DNA_SIZE = 13 # 5000 < 2^13
POP_SIZE = 50
MUTATE_RATE = 0.003
CROSS_RATE = 0.8
Weights = [2**N for N in range(12, -1, -1)]  

def getFitness(pred):
    return pred - np.min(pred)

def transDNA(DNA):
    return -1 + 7 / (2**13 - 1) * (DNA*Weights).sum(axis=1)

def select(pop, fitness):
    #print(fitness)
    index = np.random.choice(POP_SIZE, POP_SIZE, p=fitness/np.sum(fitness))
    return pop[index]

def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        index = np.random.randint(POP_SIZE)
        cross = np.random.randint(2, size=DNA_SIZE)
        parent[cross] = pop[index, cross]
    return parent

def mutate(child):
    for i in range(DNA_SIZE):
        if np.random.rand() < MUTATE_RATE:
            child[i] = int(not child[i])
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))

plt.ion()
#plt.figure()
plt.plot(x, F(x))

for _ in range(200):
    predx = transDNA(pop)
    predy = F(predx)
    predy[np.isnan(predy)] = 0
    
    if 'sca' in globals():
        sca.remove()
    sca = plt.scatter(predx, predy, c='r', lw=0.8)
    plt.pause(0.1)
    
    fitness = getFitness(predy)
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child

plt.ioff()
plt.show()

