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
    returns:
        weights     numpy array         weights for this model
        bias        float               bias for this model
    """
    def TrainEpoch(self, examples, labels):
        self.initialize()
        while True:
            # shuffle the data
            idxs = range(len(examples))
            random.shuffle(idxs)
            shuffled = [(examples[idx], labels[idx]) for idx in idxs]
            # TrainIter returns True if achieves 100 percent on data
            if self.TrainIter(shuffled):
                return self.w, self.b

    """
    Perceptron.TrainIter(self, examples)
    trains all examples once
    parameters:
        examples    [numpy arrays]      a list of feature vectors to train on
    returns:
        correct     Bool                True if all examples correctly classified
                                        False otherwise
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

    """
    Perceptron.TrainExample(self, example)
    trains single example
    parameters:
        example    numpy array          a feature vector to train on
    returns:
        updated     Bool                whether weights/bias updated
        weights     numpy array         weights for this model
        bias        float               bias for this model
    """
    def TrainExample(self, x, y):
        # test the model
        if self.Classify(x) * y <= 0:
            # mis-classified
            self.w += y * x
            self.b += y
            return True, self.w, self.b
        else:
            return False, self.w, self.b

    def Classify(self, example):
        return np.dot(self.w, example) + self.b

    def Test(self, examples, weights=None, bias=None):
        if weights is not None and bias is not None:
            self.w = weights
            self.b = bias
        return [self.Classify(x) for x in examples]

    def SetWeightBias(self, weights, bias):
        self.w = weights
        self.b = bias