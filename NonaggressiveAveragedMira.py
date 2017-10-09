import numpy as np
from perceptron import *


class NonaggressiveAveragedMira(Perceptron):
    def __init__(self, dimension):
        Perceptron.__init__(self, dimension)
        self.wavg = np.zeros(dimension)
        self.bavg = 0
        self.c = 0

    def initialize(self):
        Perceptron.initialize(self)
        self.wavg = np.zeros(self.dimension)
        self.bavg = 0
        self.c = 0

    def TrainEpoch(self, examples, labels):
        self.initialize()
        count = 0.0
        while True:
            count += 1
            idxs = range(len(examples))
            random.shuffle(idxs)
            shuffled = [(examples[idx], labels[idx]) for idx in idxs]
            stop = self.TrainIter(shuffled)
            self.wavg += self.w
            self.bavg += self.b
            if stop:
                return self.wavg / count, self.b / count

    def TrainExample(self, x, y):
        self.c += 1
        if y * Perceptron.Classify(self, x) <= 0:
            deno =abs(np.dot(x, x))
            learning_rate = (y - np.dot(self.w,x))/(deno**2)
            self.w = self.w + (learning_rate * x)
            self.b = self.b + learning_rate
            self.wavg += learning_rate * x * self.c
            self.bavg += learning_rate * self.c
            w, b = self.GetWeightsBias()
            return True, w, b
        else:
            w, b = self.GetWeightsBias()
            return False, w, b

    def Test(self, examples, weights=None, bias=None):
        if weights is None or bias is None:
            w, b = self.GetWeightsBias()
        else:
            w, b = weights, bias
        return [np.dot(w, xi) + b for xi in examples]

    def GetWeightsBias(self):
        return self.w - self.wavg / self.c, self.b - self.bavg / self.c
