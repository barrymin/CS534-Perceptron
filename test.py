#!/bin/python

from data import *
from perceptron import *

# preprocess
train_data = read_examples('income-data/income.train.txt')
train_data = [(d[:-1], d[-1]) for d in train_data]
train_examples, train_labels = [x for x, y in train_data], [y for x, y in train_data]
emb, rev_emb = embed_data(train_examples)
train_binarized_features = binarize(train_examples, emb)

dev_data = read_examples('income-data/income.dev.txt')
dev_data = [(d[:-1], d[-1]) for d in dev_data]
dev_examples, dev_labels = [x for x, y in dev_data], [y for x, y in dev_data]
dev_binarized_features = binarize(dev_examples, emb)

# train
p = Perceptron(dimension=len(train_binarized_features[0]))
p.TrainEpoch(train_binarized_features, train_labels)

# test
classified = p.Test(dev_binarized_features)
err_count = 0
for i, l in enumerate(classified):
    if l * dev_labels[i] <= 0.:
        err_count += 1
        print ('Misclassified: {} \n\tas {}'.format(dev_examples[i], l))
print('\n{} mistakes out of {} dev set examples\n'.format(err_count, len(dev_examples)))