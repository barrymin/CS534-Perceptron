#!/bin/python

from data import *
from perceptron import *

# preprocess
train_data = read_examples('income-data/income.train.txt')
train_data = [(d[:-1], d[-1]) for d in train_data]
examples, labels = [x for x, y in train_data], [y for x, y in train_data]
emb, rev_emb = embed_data(examples)
binarized_features = binarize(examples, emb)

# train
p = Perceptron(dimension=len(binarized_features[0]))
p.TrainEpoch(binarized_features, labels)

# test (on training set for now...)
c = p.Test(binarized_features)
err_count = 0
for i, l in enumerate(c):
    if l * labels[i] <= 0.:
        err_count += 1
        print ('Misclassified: {} \n\tas {}'.format(examples[i], l))
print('\n{} mistakes out of {} examples\n'.format(err_count, len(examples)))