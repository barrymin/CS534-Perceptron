#!/bin/python

import random
import numpy as np

class Perceptron:
    def __init__(self, dimension):
        self.dimension = dimension
        self.initialize()

    def initialize(self):
        self.w = np.zeros(self.dimension)
        self.b = 0.

    """
    Perceptron.Train(examples)
    trains the perceptron on examples provided
    parameters:
    examples    [numpy arrays]      a list of feature vectors to train on
    labels      [float]             a list of ground-truth labels, same length
    """
    def TrainEpoch(self, examples, labels):
        self.initialize()
        count = 0
        while True:
            count += 1
            print('iter {}'.format(count))
            idxs = range(len(examples))
            random.shuffle(idxs)
            shuffled = [(examples[idx], labels[idx]) for idx in idxs]
            if self.TrainIter(shuffled):
                # 100 percent on training data: done!
                return self.w, self.b
    """
    Perceptron.TrainIter(self, examples)
    trains all examples once
    parameters:
    examples    [numpy arrays]      a list of feature vectors to train on
    """
    def TrainIter(self, examples):
        correct = True
        for x, y in examples:
            # test the model
            if self.Classify(x) * y <= 0:
                # mis-classified
                correct = False
                self.w += y * x
                self.b += y
        return correct

    def Classify(self, example):
        return np.dot(self.w, example) + self.b

    def Test(self, examples):
        return [self.Classify(x) for x in examples]

