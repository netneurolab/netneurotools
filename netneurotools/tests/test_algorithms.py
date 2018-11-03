# -*- coding: utf-8 -*-

import numpy as np
from netneurotools import algorithms

rs = np.random.RandomState(1234)


def test_zrand():
    # make the same two-group community assignments (with different labels)
    label = np.ones((100, 1))
    X, Y = np.vstack((label, label * 2)), np.vstack((label * 2, label))
    # compare
    assert algorithms.zrand(X, Y) == algorithms.zrand(X, Y[::-1])
    random = rs.choice([0, 1], size=X.shape)
    assert algorithms.zrand(X, Y) > algorithms.zrand(X, random)
    assert algorithms.zrand(X, Y) == algorithms.zrand(X[:, 0], Y[:, 0])


def test_zrand_partitions():
    # make random communities
    comm = rs.choice(range(6), size=(10, 100))
    all_diff = algorithms.zrand_partitions(comm)
    all_same = algorithms.zrand_partitions(np.repeat(comm[:, [0]], 10, axis=1))

    # partition of labels that are all the same should have higher average
    # zrand and lower stdev zrand
    assert all_same[0] > all_diff[0]
    assert all_same[1] < all_diff[1]
