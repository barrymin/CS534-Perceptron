import numpy as np
from perceptron import *


class Naive_perceptron(Perceptron):
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
            self.w += self.learning_rate * y * x
            self.b += self.learning_rate * y
            updated = True
        else:
            updated = False
        self.wavg += self.w
        self.bavg += self.b
        w, b = self.GetWeightsBias()
        return updated, w, b

    def Test(self, examples, weights=None, bias=None):
        if weights is None or bias is None:
            w, b = self.GetWeightsBias()
        else:
            w, b = weights, bias
        return [np.dot(w, xi) + b for xi in examples]

    def GetWeightsBias(self):
        return self.wavg / self.c, self.bavg / self.c
