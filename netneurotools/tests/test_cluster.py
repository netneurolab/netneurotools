# -*- coding: utf-8 -*-
"""
For testing netneurotools.cluster functionality
"""

import numpy as np
import pytest

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
    assert np.allclose(cluster.match_cluster_labels(c1, c2), out)


@pytest.mark.xfail
def test_match_assignments(assignments):
    assert False


@pytest.mark.xfail
def test_reorder_assignments(assignments):
    assert False


@pytest.mark.xfail
def test_find_consensus(assignments):
    assert False
