#!/bin/python

from __future__ import division
from data import *
from perceptron import *
from naive_perceptron import *
import sys

MAX_EPOCHS = 5

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

test_data = read_examples('income-data/income.test.txt')
test_examples = [d[:-1] for d in test_data]
test_binarized_features = binarize(test_examples, emb)

p = Naive_perceptron(dimension=len(train_binarized_features[0]))
# p = Perceptron(dimension=len(train_binarized_features[0]))


def test(inputs, xs, ys, weights, bias):
    classified = p.Test(xs, weights, bias)
    err_count = 0
    for i, l in enumerate(classified):
        if l * ys[i] <= 0.:
            err_count += 1
            # print ('Misclassified: {} \n\tas {}'.format(inputs[i], l))
    return err_count


# train the model
count = 0
epochs = 1
min_err = len(dev_labels)
best_w, best_b = None, None
while epochs < MAX_EPOCHS:
    count = 0
    epochs += 1
    # print('\nEpoch {}'.format(epochs))

    # shuffle the data
    idxs = range(len(train_binarized_features))
    random.shuffle(idxs)
    shuffled = [(train_binarized_features[idx], train_labels[idx]) for idx in idxs]

    # run through the training set
    correct = True
    for i, x in enumerate(train_binarized_features):
        count += 1
        updated, w, b = p.TrainExample(x, train_labels[i])
        if count % 1000 == 0:
            # every 1000 examples, test on dev set
            misclassified = test(dev_examples, dev_binarized_features, dev_labels, None, None)
            print >> sys.stderr, 'Epoch {:.3}\t{:.4}'.format(
                epochs + count / len(train_labels), misclassified / len(dev_labels)
            )
            if misclassified < min_err:
                # best result so far => update best model
                min_err = misclassified
                # best_w, best_b = w, b
                best_w = w.tolist()
                best_b = b

        if updated:
            correct = False

    if correct:
        print('\nTraining complete. Convergence required {} epochs.\n'.format(epochs))
        break

# test
print >> sys.stderr, '\nTesting on dev set:'

# p.w = np.array(best_w)
# p.b = best_b

errs = test(dev_examples, dev_binarized_features, dev_labels, np.array(best_w), best_b)
print >> sys.stderr, '{} mistakes out of {} examples. '.format(errs, len(dev_labels))
print >> sys.stderr, 'Best error rate on dev set:\t{:.6}\n'.format(errs / len(dev_labels))

# label the test set
print '\nClassifier labels for test set:\n'
for i in range(len(test_data)):
    classified = p.Classify(test_binarized_features[i])
    inc = '>=50k' if classified > 0 else '<50k'
    print '{}\t{}'.format(test_data[i], inc)
