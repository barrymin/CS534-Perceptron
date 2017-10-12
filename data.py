#!/bin/python

# from __future__ import print_function
from collections import defaultdict
import numpy as np


def read_examples(path):
    data = []
    with open(path, 'r') as fp:
        for i, line in enumerate(fp.readlines()):
            fields = line.strip().split((','))
            while '' in fields:
                fields.remove('')
            if len(fields) < 9:
                print 'warning: data file line {} has unexpected format'.format(i)
                continue
            if len(fields) == 9:
                age, emp, edu, mar, job, eth, sex, hrs, nat = fields
                inc = None
            else:
                age, emp, edu, mar, job, eth, sex, hrs, nat, inc = fields
                inc = 1. if inc == ' >50K' else -1.

            data.append((int(age), emp, edu, mar, job, eth, sex, int(hrs), nat, inc))

        return data


"""
produce the feature mapping
"""
def embed_data(data):
    fields = [f for f in data[0]]
    val2num, num2val, accums = [{} for f in fields], [{} for f in fields], [0 for f in fields]
    for d in data:
        for j in range(len(d)):
            val = d[j]
            emb = val2num[j]
            rev_emb = num2val[j]
            num = accums[j]
            if val not in emb:
                emb[val] = num
                rev_emb[num] = val
                accums[j] += 1

    for i, f in enumerate(fields):
        if type(f) is int:
            minval, maxval = 0, max(val2num[i].keys()) + 20
            val2num[i] = {n : n for n in range(minval, maxval + 1)}
            num2val[i] = {n : '{}_{}'.format(i, n) for n in range(minval, maxval + 1)}

    return val2num, num2val


"""
produce the feature mapping with numeric features left as single dimensions
"""
def embed_data_numeric(data):
    fields = [f for f in data[0]]
    val2num, num2val, accums = [{} for f in fields], [{} for f in fields], [0 for f in fields]
    for d in data:
        for j in range(len(d)):
            val = d[j]
            if type(val) is int:
                continue
            emb = val2num[j]
            rev_emb = num2val[j]
            num = accums[j]
            if val not in emb:
                emb[val] = num
                rev_emb[num] = val
                accums[j] += 1
    return val2num, num2val


def embed_data_binned(data):
    fields = [f for f in data[0]]
    val2num, num2val, accums = [{} for f in fields], [{} for f in fields], [0 for f in fields]
    for d in data:
        for j in range(len(d)):
            val = d[j]
            emb = val2num[j]
            rev_emb = num2val[j]
            num = accums[j]
            if val not in emb:
                emb[val] = num
                rev_emb[num] = val
                accums[j] += 1

    for i, f in enumerate(fields):
        if type(f) is int:
            minval, maxval = 0, max(val2num[i].keys()) + 20
            num_bins = maxval / 15
            for j in range(maxval):
                val2num[i][j] = j / 15
                num2val[i][j / 15] = 'range {}-{}'.format(j % 15, j % 15 + 15)

    return val2num, num2val


def binarize(data, embedding):
    dimension_sizes = [len(f) for f in embedding]
    processed = []
    labels = [None for d in data]
    for i, d in enumerate(data):
        x = np.zeros(sum(dimension_sizes))
        offset = 0
        for j, fd in enumerate(dimension_sizes):
            val = embedding[j][d[j]]
            x[offset + val] = 1.
            offset += fd

        processed.append(x)

    return processed


def binarize_except_numeric(data, embedding):
    dimension_sizes = [len(f) for f in embedding]
    processed = []
    labels = [None for d in data]
    for i, d in enumerate(data):
        x = np.zeros(sum(dimension_sizes) + 2)
        offset = 0
        for j, size in enumerate(dimension_sizes):
            if size == 0:
                offset += 1
                x[offset] = int(d[j])
            else:
                val = embedding[j][d[j]]
                x[offset + val] = 1.
                offset += size

        processed.append(x)

    return processed


def binarize_binned(data, embedding):
    dimension_sizes = []
    for emb in embedding:
        if type(emb.keys()[0]) is int:
            dimension_sizes.append(max(emb.values()))
        else:
            dimension_sizes.append(len(emb))
    processed = []
    labels = [None for d in data]
    for i, d in enumerate(data):
        x = np.zeros(sum(dimension_sizes))
        offset = 0
        for j, size in enumerate(dimension_sizes):
            val = embedding[j][d[j]]
            x[offset + val] = 1.
            if type(val) is int:
                offset += max(embedding[j].values())
            else:
                offset += size

        processed.append(x)

    return processed