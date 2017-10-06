#!/bin/python

from data import *
from perceptron import *

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

p = Perceptron(dimension=len(train_binarized_features[0]))


def test(inputs, xs, ys):
    classified = p.Test(xs)
    err_count = 0
    for i, l in enumerate(classified):
        if l * ys[i] <= 0.:
            err_count += 1
            # print ('Misclassified: {} \n\tas {}'.format(inputs[i], l))
    print('{} mistakes out of {} examples'.format(err_count, len(inputs)))
    return err_count


# train the model
count = 0
epochs = 0
min_err = len(dev_labels)
best_w, best_b = None, None
while epochs < MAX_EPOCHS:
    epochs += 1
    print('\nEpoch {}'.format(epochs))
    # shuffle the data
    idxs = range(len(train_binarized_features))
    random.shuffle(idxs)
    shuffled = [(train_binarized_features[idx], train_labels[idx]) for idx in idxs]
    correct = True

    # run through the training set
    for i, x in enumerate(train_binarized_features):
        count += 1
        updated, w, b = p.TrainExample(x, train_labels[i])
        if count % 1000 == 0:
            # every 1000 examples, test on dev set
            misclassified = test(dev_examples, dev_binarized_features, dev_labels)
            if misclassified < min_err:
                # best result so far => update best model
                min_err = misclassified
                best_w, best_b = w, b

        if updated:
            correct = False

    if correct:
        print('\nTraining complete. Convergence required {} epochs.\n'.format(epochs))
        break



# p.TrainEpoch(train_binarized_features, train_labels)


# test
print('\nTesting on dev set:')
p.w = best_w
p.b = best_b
test(dev_examples, dev_binarized_features, dev_labels)
print('\n')

# label the test set
for i in range(len(dev_data)):
    classified = p.Classify(dev_binarized_features[i])
    inc = '>=50k' if classified > 0 else '<50k'
    print('{}\t{}'.format(dev_data[i], inc))
