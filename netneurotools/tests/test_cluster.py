# -*- coding: utf-8 -*-
"""
For testing netneurotools.cluster functionality
"""

import bct
import numpy as np
import pytest
from sklearn.cluster import k_means, spectral_clustering

from netneurotools import cluster


@pytest.mark.parametrize('c1, c2, out', [
    # uniform communities
    (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
     np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
     np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])),
    # perfectly matched but mislabeled communities
    (np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
     np.array([2, 2, 2, 1, 1, 1, 3, 3, 3]),
     np.array([2, 2, 2, 1, 1, 1, 3, 3, 3])),
    # 1 cluster --> 2 clusters
    (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
     np.array([1, 1, 1, 1, 2, 2, 2, 2, 2]),
     np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])),
    # 3 clusters --> 2 clusters
    (np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
     np.array([1, 1, 1, 1, 1, 2, 2, 2, 2]),
     np.array([1, 1, 1, 3, 3, 3, 2, 2, 2]))
])
def test_match_cluster_labels(c1, c2, out):
    assert np.all(cluster.match_cluster_labels(c1, c2) == out)


def test_match_assignments():
    # generate some random data to be clustered (must be symmetric)
    rs = np.random.RandomState(1234)
    data = rs.rand(100, 100)
    data = (data + data.T) / 2

    # cluster it 200 times (spectral clustering provides unstable labels but
    # should provide ~stable(ish) assignments)
    assignments = [spectral_clustering(data, n_clusters=2, random_state=rs)
                   for n in range(200)]
    assignments = np.column_stack(assignments)

    # make sure the labels are, in fact, unstable
    assert not np.all(assignments[:, [0]] == assignments)

    # match labels and assert that we got perfect matches (this is not 100%
    # guaranteed with spectral clustering but it is...pretty likely)
    matched = cluster.match_assignments(assignments, seed=rs)
    assert np.all(matched[:, [0]] == matched)

    # check that we didn't _actually_ change cluster assignments with matching;
    # the agreement matrices should match!
    assert np.allclose(bct.clustering.agreement(assignments),
                       bct.clustering.agreement(matched))


def test_reorder_assignments():
    # generate a bunch of ~random(ish) clustering assignments that have a bit
    # of consistency but aren't all identical
    rs = np.random.RandomState(1234)
    data = rs.rand(100, 200)
    assignments = [k_means(data, n_clusters=2)[1] for n in range(200)]
    assignments = np.column_stack(assignments)

    # make sure the labels are quite unstable
    assert not np.all(assignments[:, [0]] == assignments)

    # re-order labels and assert that we still _don't_ have perfect matches
    # (we're re-labelling the matrix but k-means does not provide stable
    # clustering assignments so we shouldn't get identical assignments even
    # after "matching")
    reordered, idx = cluster.reorder_assignments(assignments, seed=1234)
    assert not np.all(reordered[:, [0]] == reordered)

    # make sure that the returned idx does exactly what it's supposed to
    matched = cluster.match_assignments(assignments, seed=1234)[idx]
    assert np.all(matched == reordered)


@pytest.mark.parametrize('assignments, clusters', [
    (np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2],
               [1, 2, 0], [1, 2, 0], [1, 2, 0],
               [2, 0, 1], [2, 0, 1], [2, 0, 1]]),
     np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))
])
def test_find_consensus(assignments, clusters):
    assert np.all(cluster.find_consensus(assignments) == clusters)
