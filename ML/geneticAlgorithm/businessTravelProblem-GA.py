#!/usr/bin/env python3
# coding : utf-8

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(10)

cities = 20
points = 10 * np.random.random_sample(size=(cities,2))
dis = np.zeros((cities,cities))

for i in range(cities):
    for j in range(i):
        dis[i,j] = dis[j,i] = np.linalg.norm(points[i]-points[j])

def checkSimiliar(func, *args, **kwargs):
    def innerFunc(self, *args, **kwargs):
        p = func(self, *args, **kwargs)
        tmp = tuple(p.tolist())
        while tmp in self.appear:
            self.appear.add(tuple(p.tolist()))
            p = func(*args, **kwargs)
            tmp = tuple(p.tolist())
        return p
    return innerFunc

np.random.seed(int(time.time()))

class GA:
    def __init__(self, DNA_SIZE=cities, POP_SIZE=5000, 
                 CROSS_RATE=0.1, MUTATE_RATE=0.5):
        self.POP_SIZE=POP_SIZE
        self.CROSS_RATE = CROSS_RATE
        self.MUTATE_RATE = MUTATE_RATE
        self.DNA_SIZE = DNA_SIZE

        series = [np.arange(self.DNA_SIZE) for _ in range(self.POP_SIZE)]
        for s in series:
            np.random.shuffle(s)
        self.pop = np.array(series)
        self.appear = set()
        # print(self.pop, self.pop.shape)
    
    def getFitness(self):
        fitness = np.zeros((self.POP_SIZE,))
        for i in range(self.POP_SIZE):
            for j in range(1, cities):
                fitness[i] += dis[self.pop[i,j],self.pop[i, j-1]]
        fitness = np.exp(self.POP_SIZE * fitness)
        return np.max(fitness) - fitness
        # return np.exp(1/fitness)
    
    def choose(self, fitness):
        index = np.random.choice(self.POP_SIZE, size=self.POP_SIZE, p=fitness/np.sum(fitness))
        # self.pop = self.pop[index]
        return self.pop[index]
    
    @checkSimiliar
    def crossover(self, parent, pop):
        if np.random.rand() < self.CROSS_RATE:
            index = np.random.randint(self.POP_SIZE)
            cross = np.random.randint(0, 2, self.DNA_SIZE).astype(np.bool)
            keep = parent[~cross]
            swap = pop[index, np.isin(pop[index], keep, invert=True)]
            parent[:] = np.concatenate((keep, swap))
        return parent
        
    def mutate(self, child):
        for i in range(self.DNA_SIZE):
            if np.random.rand() < self.MUTATE_RATE:
                swap = np.random.randint(self.DNA_SIZE)
                a, b = child[swap], child[i]
                child[i], child[swap] = a, b
        return child

    def evolve(self):
        fitness = self.getFitness()
        pop = self.choose(fitness)
        pop_copy = pop.copy()
        for j in range(self.POP_SIZE):
            child = self.crossover(pop[j], pop_copy)
            child = self.mutate(child)
            pop[j][:] = child 
        self.pop = pop
    
    def getCost(self, pop=None):
        if pop is not None:
            return sum([dis[pop[i-1],pop[i]] for i in range(1, cities)])
        else:
            return [sum([dis[pop[j,i-1],pop[j,i]] for i in range(1, cities)]) for j in range(self.POP_SIZE)]
        
    def getBestSeries(self):
        return self.pop[np.argmax(self.getFitness())]

def plotConnection(pop):
    plt.cla()
    plt.scatter(points[:,0], points[:,1], c='k', s=80)
    cost = 0
    for i in range(1, cities):
        cost += dis[pop[i-1],pop[i]]
        plt.plot(points[[pop[i-1],pop[i]],0], points[[pop[i-1],pop[i]],1], c='r')
    plt.title("using cost : %.2f" % cost)
    plt.pause(0.08)

ga = GA()
plt.ion()

series = ga.getBestSeries()
bestSolution = series, ga.getCost(series)

for i in range(2000):
    series = ga.getBestSeries()
    solution = series, ga.getCost(series)
    if solution[1] < bestSolution[1]:
        bestSolution = solution

    if i % 200 == 0:
        plotConnection(bestSolution[0])
        print("%d-th iteration..." % (i + 1))
    ga.evolve()

print(bestSolution)
