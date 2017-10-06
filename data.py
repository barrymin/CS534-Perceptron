#!/bin/python

from __future__ import print_function
from collections import defaultdict
import numpy as np


def read_examples(path):
    data = []
    with open(path, 'r') as fp:
        for i, line in enumerate(fp.readlines()):
            fields = line.split((', '))
            if len(fields) < 9:
                print('warning: data file line {} has unexpected format'.format(i))
                continue
            if len(fields) == 9:
                age, emp, edu, mar, job, eth, sex, hrs, nat = fields
                inc = None
            else:
                age, emp, edu, mar, job, eth, sex, hrs, nat, inc = fields
                inc = 1. if inc == '>50K' else -1.

            data.append((int(age), emp, edu, mar, job, eth, sex, int(hrs), nat, inc))

        return data


"""
stupid function to count how many values each field has
"""
def embed_data(data):
    fields = [f for f in data[0]]
    val2num, num2val, accums = [{} for f in fields], [{} for f in fields], [0 for f in fields]
    for d in data:
        for j in range(len(d) - 1):
            # for j, x_j in enumerate(d):
            #     if j == len(d) - 1:
            #         continue
            emb = val2num[j]
            rev_emb = num2val[j]
            val = d[j]
            num = accums[j]
            if val not in emb:
                emb[val] = num
                rev_emb[num] = val
                # num = emb[val]
                accums[j] += 1

    return val2num, num2val


def binarize(data, embedding):
    dimension_sizes = [len(f) for f in embedding]
    processed = []
    labels = [None for d in data]
    for i, d in enumerate(data):
        x = np.zeros(sum(dimension_sizes))

        offset = 0
        for j, fd in enumerate(dimension_sizes):
            if j == len(dimension_sizes) - 1:
                labels[i] = d[j]
            else:
                val = embedding[j][d[j]]
                x[offset + val] = 1.
                offset += fd

        processed.append(x)

    return processed

