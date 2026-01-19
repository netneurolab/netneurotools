"""For testing netneurotools.modularity.modules functionality."""

import pytest
import numpy as np
from sklearn.cluster import k_means, spectral_clustering

from netneurotools import modularity

rs = np.random.RandomState(1234)


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
    """Test matching of cluster labels."""
    assert np.all(modularity.match_cluster_labels(c1, c2) == out)


def test_match_assignments():
    """Test matching of clustering assignments."""
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
    matched = modularity.match_assignments(assignments, seed=rs)
    assert np.all(matched[:, [0]] == matched)

    # check that we didn't _actually_ change cluster assignments with matching;
    # the agreement matrices should match!
    assert np.allclose(modularity.agreement_matrix(assignments),
                       modularity.agreement_matrix(matched))


def test_reorder_assignments():
    """Test re-ordering of clustering assignments."""
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
    reordered, idx = modularity.reorder_assignments(assignments, seed=1234)
    assert not np.all(reordered[:, [0]] == reordered)

    # make sure that the returned idx does exactly what it's supposed to
    matched = modularity.match_assignments(assignments, seed=1234)[idx]
    assert np.all(matched == reordered)


@pytest.mark.parametrize('assignments, clusters', [
    (np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2],
               [1, 2, 0], [1, 2, 0], [1, 2, 0],
               [2, 0, 1], [2, 0, 1], [2, 0, 1]]),
     np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))
])
def test_find_consensus(assignments, clusters):
    """Test finding consensus clustering."""
    assert np.all(modularity.find_consensus(assignments) == clusters)


def test_agreement_matrix():
    """Test calculation of agreement matrix."""
    assignments = np.array([[0, 0],
                            [0, 1],
                            [1, 1]])
    expected = np.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]])
    agreement = modularity.agreement_matrix(assignments)
    assert np.all(agreement == expected)


def test_zrand():
    """Test calculation of zrand."""
    # make the same two-group community assignments (with different labels)
    label = np.ones((100, 1))
    X, Y = np.vstack((label, label * 2)), np.vstack((label * 2, label))
    # compare
    assert modularity.zrand(X, Y) == modularity.zrand(X, Y[::-1])
    random = rs.choice([0, 1], size=X.shape)
    assert modularity.zrand(X, Y) > modularity.zrand(X, random)
    assert modularity.zrand(X, Y) == modularity.zrand(X[:, 0], Y[:, 0])


def test_zrand_partitions():
    """Test calculation of zrand for partitions."""
    # make random communities
    comm = rs.choice(range(6), size=(10, 100))
    all_diff = modularity._zrand_partitions(comm)
    all_same = modularity._zrand_partitions(np.repeat(comm[:, [0]], 10, axis=1))

    # partition of labels that are all the same should have higher average
    # zrand and lower stdev zrand
    assert np.nanmean(all_same) > np.nanmean(all_diff)
    assert np.nanstd(all_same) < np.nanstd(all_diff)
