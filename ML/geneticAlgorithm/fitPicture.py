#!/usr/bin/env python3
# coding : utf-8

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

file = '/path/to/yourfile'
pic = cv2.imread(file, 0)
print(pic.shape)

class GA:
    def __init__(self, DNA_SIZE=pic.shape, POP_SIZE=50, 
                 CROSS_RATE=0.8, MUTATE_RATE=0.003):
        self.POP_SIZE=POP_SIZE
        self.CROSS_RATE = CROSS_RATE
        self.MUTATE_RATE = MUTATE_RATE
        self.DNA_SIZE = DNA_SIZE
        self.pop = np.random.randint(256, size=(*self.DNA_SIZE, self.POP_SIZE))
    
    def getFitness(self):
        fitness = np.zeros(self.POP_SIZE)
        for i in range(self.POP_SIZE):
            fitness[i] = np.sum(255 - abs(pic-self.pop[:,:,i]))
        return fitness
    
    def select(self, fitness):
        #if np.random.rand() < self.CROSS_RATE:
        index = np.random.choice(self.POP_SIZE, self.POP_SIZE, p=fitness/fitness.sum())
        self.pop = self.pop[:,:,index]
        return self.pop

    def crossover(self, parent):
        if np.random.rand() < self.CROSS_RATE:
            index = np.random.randint(self.POP_SIZE)
            cross = np.random.randint(2, size=self.DNA_SIZE).astype(np.bool)
            parent[cross] = self.pop[:,:,index][cross]
        return parent
        
    def mutate(self, child):
        if np.random.rand() < self.MUTATE_RATE:
            mask = np.random.randint(2, size=self.DNA_SIZE).astype(np.bool)
            child[mask] = pic[mask]
        return child
            
    def showPic(self):
        plt.cla()
        for i in range(self.POP_SIZE):
            plt.subplot(3,3,i+1)
            plt.imshow(self.pop[:,:,i])
            plt.pause(0.2)

    def savePic(self, count=None):
        if count is None:
            cv2.imwrite("./GAPic/pop-%d-best.jpg" % int(time.time()), 
                    self.pop[:,:,np.argmin(self.getFitness())])
        else:
            for i in np.random.randint(self.POP_SIZE, size=(count,)):
                cv2.imwrite("./GAPic/pop-%d-%d.jpg" % 
                    (int(time.time()), i), self.pop[:,:,i])


ga = GA()
for i in range(300):
    # ga.showPic()
    fitness = ga.getFitness()

    pop = ga.select(fitness)
    pop_copy = pop.copy()
    for j in range(ga.POP_SIZE):
        parent = ga.pop[:,:,j]
        child = ga.crossover(parent)
        child = ga.mutate(child)
        parent[:] = child

    print("%s-th iteration...." % str(i+1))
    if i % 10 == 0:
        ga.savePic()





