import numpy as np
from perceptron import *

class Naive_perceptron(Perceptron):
    def __init__(self, dimension):
        Perceptron.__init__(self,dimension)
        self.wavg = np.zeros(dimension)
        self.bavg = 0
        
    def TrainEpoch(self,examples,labels):
        self.wavg = np.zeros(self.dimension)
        self.w = np.zeros(self.dimension)
        self.bavg = 0
        count = 0.0
        while True:
            count+=1
            idxs = range(len(examples))
            random.shuffle (idxs)
            shuffled = [(examples[idx], labels[idx]) for idx in idxs]
            stop = self.TrainIter(shuffled)
            self.wavg += self.w
            self.bavg += self.b
            if stop:
                return self.wavg/count, self.b/count
                
        