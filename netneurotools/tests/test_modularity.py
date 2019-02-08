# -*- coding: utf-8 -*-

import numpy as np

from netneurotools import modularity

rs = np.random.RandomState(1234)


def test_dummyvar():
    # generate small example dummy variable code
    out = modularity._dummyvar(np.array([1, 1, 2, 3, 3]))
    assert np.all(out == np.array([[1, 0, 0],
                                   [1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1],
                                   [0, 0, 1]]))

    allones = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    assert np.all(modularity._dummyvar(allones) == allones)


def test_zrand():
    # make the same two-group community assignments (with different labels)
    label = np.ones((100, 1))
    X, Y = np.vstack((label, label * 2)), np.vstack((label * 2, label))
    # compare
    assert modularity.zrand(X, Y) == modularity.zrand(X, Y[::-1])
    random = rs.choice([0, 1], size=X.shape)
    assert modularity.zrand(X, Y) > modularity.zrand(X, random)
    assert modularity.zrand(X, Y) == modularity.zrand(X[:, 0], Y[:, 0])


def test_zrand_partitions():
    # make random communities
    comm = rs.choice(range(6), size=(10, 100))
    all_diff = modularity._zrand_partitions(comm)
    all_same = modularity._zrand_partitions(np.repeat(comm[:, [0]], 10,
                                                      axis=1))

    # partition of labels that are all the same should have higher average
    # zrand and lower stdev zrand
    assert np.nanmean(all_same) > np.nanmean(all_diff)
    assert np.nanstd(all_same) < np.nanstd(all_diff)
