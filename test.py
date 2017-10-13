#!/bin/python

from __future__ import division
from data import *
import sys
from perceptron import *
from naive_perceptron import *
from smart_perceptron import *
from NonaggressiveDefaultMira import *
from NonaggressiveAveragedMira import *
from AggressiveAveragedMira import *
from AggressiveDefaultMira import *
from vlr_perceptron import *

MAX_EPOCHS = 5
SORT = False
NUMERIC_FEATURES = False
BINNED = False
NORMALIZE_DATA = True

def normalize(feature):
    fmean = feature.mean()
    fstd = np.std(feature)
    temp = np.ones(np.size(feature))
    feature = (feature - temp * fmean) / fstd
    return feature

# preprocess
if BINNED:
    # numeric features binned
    train_data = read_examples('income-data/income.train.txt')
    train_data = [(d[:-1], d[-1]) for d in train_data]
    train_examples, train_labels = [x for x, y in train_data], [y for x, y in train_data]

    dev_data = read_examples('income-data/income.dev.txt')
    dev_data = [(d[:-1], d[-1]) for d in dev_data]
    dev_examples, dev_labels = [x for x, y in dev_data], [y for x, y in dev_data]
    dev_binarized_features = binarize_binned(dev_examples, emb)

    test_data = read_examples('income-data/income.test.txt')
    test_examples = [d[:-1] for d in test_data]
    test_binarized_features = binarize_binned(test_examples, emb)
elif NUMERIC_FEATURES:
    # do the same thing with numeric features intact
    train_data = read_examples('income-data/income.train.txt')
    train_data = [(d[:-1], d[-1]) for d in train_data]
    train_examples, train_labels = [x for x, y in train_data], [y for x, y in train_data]
    emb, rev_emb = embed_data_numeric(train_examples)
    train_binarized_features = binarize_except_numeric(train_examples, emb)

    dev_data = read_examples('income-data/income.dev.txt')
    dev_data = [(d[:-1], d[-1]) for d in dev_data]
    dev_examples, dev_labels = [x for x, y in dev_data], [y for x, y in dev_data]
    dev_binarized_features = binarize_except_numeric(dev_examples, emb)

    test_data = read_examples('income-data/income.test.txt')
    test_examples = [d[:-1] for d in test_data]
    test_binarized_features = binarize_except_numeric(test_examples, emb)
else:
    # all features binarized
    train_data = read_examples('income-data/income.train.txt')
    if SORT:
        train_data = sorted(train_data, key=lambda x: x[-1], reverse=True)

    train_data = [(d[:-1], d[-1]) for d in train_data]
    train_examples, train_labels = [x for x, y in train_data], [y for x, y in train_data]

    dev_data = read_examples('income-data/income.dev.txt')
    dev_data = [(d[:-1], d[-1]) for d in dev_data]
    dev_examples, dev_labels = [x for x, y in dev_data], [y for x, y in dev_data]

    test_data = read_examples('income-data/income.test.txt')
    test_examples = [d[:-1] for d in test_data]

    if NORMALIZE_DATA:
        all_examples = train_examples + dev_examples + test_examples
        idxs = [i for i, x in enumerate( all_examples[0] ) if type( x ) is int]
        columns = [None for field in all_examples[0]]
        normalized_columns = []
        for i in idxs:
            columns[i] = [t[i] for t in all_examples]
        for i, c in enumerate(columns):
            if c is not None:
                columns[i] = normalize(np.array(c))
        normalized_examples = []
        for j, t in enumerate( all_examples ):
            example = []
            for i, field in enumerate( t ):
                if i in idxs:
                    example.append( columns[i][j] )
                else:
                    example.append( field )
            normalized_examples.append( tuple(example) )
        all_examples = normalized_examples
        emb, rev_emb = embed_data( all_examples )
        train_examples = all_examples[0:len(train_examples)]
        dev_examples = all_examples[len(train_examples):len(train_examples) + len(dev_examples)]
        test_examples = all_examples[-len(test_examples):]
    else:
        emb, rev_emb = embed_data(train_examples)


    train_binarized_features = binarize(train_examples, emb)

    dev_binarized_features = binarize(dev_examples, emb)

    test_binarized_features = binarize(test_examples, emb)

# create our perceptron
p = Perceptron(dimension=len(train_binarized_features[0]))
# p = Smart_perceptron(dimension=len(train_binarized_features[0]))
# p = Naive_perceptron(dimension=len(train_binarized_features[0]))
# p = NonaggressiveDefaultMira(dimension=len(train_binarized_features[0]))
# p = NonaggressiveAveragedMira(dimension=len(train_binarized_features[0]))
# p = AggressiveDefaultMira(dimension=len(train_binarized_features[0]), p=0.5)
# p = VariableLearningRatePerceptron(dimension=len(train_binarized_features[0]))


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
epochs = 0
min_err = len(dev_labels)
best_w, best_b, best_epoch = None, None, None
learning_rate = 1.
decay = 1.75

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
        cur_epoch = epochs + i / len( train_binarized_features )
        # p.learning_rate = 1. / (1. + decay * (cur_epoch - 1.))
        updated, w, b = p.TrainExample(x, train_labels[i])
        if count % 1000 == 0:
            print 'epoch {} lr: {}'.format( cur_epoch, p.learning_rate )
            # every 1000 examples, test on dev set
            misclassified = test(dev_examples, dev_binarized_features, dev_labels, None, None)
            print >> sys.stderr, 'Epoch {:.3}\t{:.4}'.format(
                cur_epoch, misclassified / len(dev_labels)
            )
            if misclassified < min_err:
                # best result so far => update best model
                min_err = misclassified
                best_epoch = cur_epoch
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
print >> sys.stderr, 'Best error rate on dev set:\t{:.6}'.format(errs / len(dev_labels))
print >> sys.stderr, 'Achieved at epoch {}\n'.format(best_epoch)

# label the test set
# print '\nClassifier labels for test set:\n'
fp = open('income-data/income.test.txt')
testlines = fp.readlines()
for i in range(len(test_data)):
    classified = p.Classify(test_binarized_features[i])
    inc = '>=50k' if classified > 0 else '<50k'
    # print '{}\t{}'.format(test_data[i], inc)
    print '{} {}'.format(testlines[i].strip(), inc)

print >> sys.stderr, '\n\nweights:'
weights, bias = p.GetWeightsBias()
feature_names = []
for i, r in enumerate(rev_emb):
    if len(r) == 0:
        feature_names.append('feature {}'.format(i))
    else:
        feature_names += [kv[1] for kv in sorted(r.items())]

ordered_idxs = [i for i in sorted(range(len(weights)), key=lambda x: weights[x])]

for i in ordered_idxs:

    print >> sys.stderr, '{}: {}'.format(feature_names[i], weights[i])