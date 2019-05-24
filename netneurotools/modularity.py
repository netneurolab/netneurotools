# -*- coding: utf-8 -*-
"""
Functions for working with network modularity
"""

import bct
import numpy as np
from sklearn.utils.validation import check_random_state
from . import cluster

try:
    from numba import njit, prange
    use_numba = True
except ImportError:
    prange = range
    use_numba = False


def consensus_modularity(adjacency, gamma=1, B='modularity',
                         repeats=250, null_func=np.mean, seed=None):
    """
    Finds community assignments from `adjacency` through consensus

    Performs `repeats` iterations of community detection on `adjacency` and
    then uses consensus clustering on the resulting community assignments.

    Parameters
    ----------
    adjacency : (N, N) array_like
        Adjacency matrix (weighted/non-weighted) on which to perform consensus
        community detection.
    gamma : float, optional
        Resolution parameter for modularity maximization. Default: 1
    B : str or (N, N) array_like, optional
        Null model to use for consensus clustering. If `str`, must be one of
        ['modularity', 'potts', 'negative_sym', 'negative_asym']. Default:
        'modularity'
    repeats : int, optional
        Number of times to repeat Louvain algorithm clustering. Default: 250
    null_func : callable, optional
        Function used to generate null model when performing consensus-based
        clustering. Must accept a 2D array as input and return a single value.
        Default: `np.mean`
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    consensus : (N,) np.ndarray
        Consensus-derived community assignments
    Q_all : array_like
        Optimized modularity over all `repeats` community assignments
    zrand_all : array_like
        z-Rand score over all pairs of `repeats` community assignment vectors

    References
    ----------
    Bassett, D. S., Porter, M. A., Wymbs, N. F., Grafton, S. T., Carlson,
    J. M., & Mucha, P. J. (2013). Robust detection of dynamic community
    structure in networks. Chaos: An Interdisciplinary Journal of Nonlinear
    Science, 23(1), 013142.
    """

    # generate community partitions `repeat` times
    comms, Q_all = zip(*[bct.community_louvain(adjacency, gamma=gamma, B=B)
                         for i in range(repeats)])
    comms = np.column_stack(comms)

    # find consensus cluster assignments across all partitoning solutions
    consensus = cluster.find_consensus(comms, null_func=null_func, seed=seed)

    # get z-rand statistics for partition similarity (n.b. can take a while)
    zrand_all = _zrand_partitions(comms)

    return consensus, np.array(Q_all), zrand_all


def _dummyvar(labels):
    """
    Generates dummy-coded array from provided community assignment `labels`

    Parameters
    ----------
    labels : (N,) array_like
        Labels assigning `N` samples to `G` groups

    Returns
    -------
    ci : (N, G) numpy.ndarray
        Dummy-coded array where 1 indicates that a sample belongs to a group
    """

    comms = np.unique(labels)

    ci = np.zeros((len(labels), len(comms)))
    for n, grp in enumerate(comms):
        ci[:, n] = labels == grp

    return ci


def zrand(X, Y):
    """
    Calculates the z-Rand index of two community assignments

    Parameters
    ----------
    X, Y : (n, 1) array_like
        Community assignment vectors to compare

    Returns
    -------
    z_rand : float
        Z-rand index

    See Also
    --------
    netneurotools_matlab/randz.m

    References
    ----------
    Amanda L. Traud, Eric D. Kelsic, Peter J. Mucha, and Mason A. Porter.
    (2011). Comparing Community Structure to Characteristics in Online
    Collegiate Social Networks. SIAM Review, 53, 526-543.
    """

    if X.ndim > 1 or Y.ndim > 1:
        if X.shape[-1] > 1 or Y.shape[-1] > 1:
            raise ValueError('X and Y must have only one-dimension each. '
                             'Please check inputs.')

    Xf = X.flatten()
    Yf = Y.flatten()

    n = len(Xf)
    indx, indy = _dummyvar(Xf), _dummyvar(Yf)
    Xa = indx.dot(indx.T)
    Ya = indy.dot(indy.T)

    M = n * (n - 1) / 2
    M1 = Xa.nonzero()[0].size / 2
    M2 = Ya.nonzero()[0].size / 2

    wab = np.logical_and(Xa, Ya).nonzero()[0].size / 2

    mod = n * (n**2 - 3 * n - 2)
    C1 = mod - (8 * (n + 1) * M1) + (4 * np.power(indx.sum(0), 3).sum())
    C2 = mod - (8 * (n + 1) * M2) + (4 * np.power(indy.sum(0), 3).sum())

    a = M / 16
    b = ((4 * M1 - 2 * M)**2) * ((4 * M2 - 2 * M)**2) / (256 * (M**2))
    c = C1 * C2 / (16 * n * (n - 1) * (n - 2))
    d = ((((4 * M1 - 2 * M)**2) - (4 * C1) - (4 * M))
         * (((4 * M2 - 2 * M)**2) - (4 * C2) - (4 * M))
         / (64 * n * (n - 1) * (n - 2) * (n - 3)))

    sigw2 = a - b + c + d
    # catch any negatives
    if sigw2 < 0:
        return 0
    z_rand = (wab - ((M1 * M2) / M)) / np.sqrt(sigw2)

    return z_rand


def _zrand_partitions(communities):
    """
    Calculates z-Rand for all pairs of assignments in `communities`

    Iterates through every pair of community assignment vectors in
    `communities` and calculates the z-Rand score to assess their similarity.

    Parameters
    ----------
    communities : (S, R) array_like
        Community assignments for `S` samples over `R` partitions

    Returns
    -------
    all_zrand : array_like
        z-Rand score over all pairs of `R` partitions of community assignments
    """

    n_partitions = communities.shape[-1]
    all_zrand = np.zeros(int(n_partitions * (n_partitions - 1) / 2))

    for c1 in prange(n_partitions):
        for c2 in prange(c1 + 1, n_partitions):
            idx = int((c1 * n_partitions) + c2 - ((c1 + 1) * (c1 + 2) // 2))
            all_zrand[idx] = zrand(communities[:, c1], communities[:, c2])

    return all_zrand


if use_numba:
    _dummyvar = njit(_dummyvar)
    zrand = njit(zrand)
    _zrand_partitions = njit(_zrand_partitions, parallel=True)


def get_modularity(adjacency, comm, gamma=1):
    """
    Calculates modularity contribution for each community in `comm`

    Parameters
    ----------
    adjacency : (N, N) array_like
        Adjacency (e.g., correlation) matrix
    comm : (N,) array_like
        Community assignment vector splitting `N` subjects into `G` groups
    gamma : float, optional
        Resolution parameter used in original modularity maximization.
        Default: 1

    Returns
    -------
    comm_q : (G,) ndarray
        Relative modularity for each community

    See Also
    --------
    netneurotools.modularity.get_modularity_z
    netneurotools.modularity.get_modularity_sig
    """

    adjacency, comm = np.asarray(adjacency), np.asarray(comm)
    s = adjacency.sum()
    B = adjacency - (gamma * np.outer(adjacency.sum(axis=1),
                                      adjacency.sum(axis=0)) / s)

    # find modularity contribution of each community
    communities = np.unique(comm)
    comm_q = np.empty(shape=communities.size)
    for n, ci in enumerate(communities):
        inds = comm == ci
        comm_q[n] = B[np.ix_(inds, inds)].sum() / s

    return comm_q


def get_modularity_z(adjacency, comm, gamma=1, n_perm=10000, seed=None):
    """
    Calculates average z-score of community assignments by permutation

    Parameters
    ----------
    adjacency : (N, N) array_like
        Adjacency (correlation) matrix
    comm : (N,) array_like
        Community assignment vector splitting `N` subjects into `G` groups
    gamma : float, optional
        Resolution parameter used in original modularity maximization.
        Default: 1
    n_perm : int, optional
        Number of permutations. Default: 10000
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    q_z : float
        Average Z-score of modularity of communities

    See Also
    --------
    netneurotools.modularity.get_modularity
    netneurotools.modularity.get_modularity_sig
    """

    rs = check_random_state(seed)

    real_qs = get_modularity(adjacency, comm, gamma)
    simu_qs = np.empty(shape=(np.unique(comm).size, n_perm))
    for perm in range(n_perm):
        simu_qs[:, perm] = get_modularity(adjacency,
                                          rs.permutation(comm),
                                          gamma)

    # avoid instances where dist.std(1) == 0
    std = simu_qs.std(axis=1)
    if std == 0:
        return np.mean(real_qs - simu_qs.mean(axis=1))
    else:
        return np.mean((real_qs - simu_qs.mean(axis=1)) / std)


def get_modularity_sig(adjacency, comm, gamma=1, n_perm=10000, alpha=0.01,
                       seed=None):
    """
    Calculates signifiance of community assignments in `comm` by permutation

    Parameters
    ----------
    adjacency : (N, N) array_like
        Adjacency (correlation) matrix
    comm : (N,) array_like
        Community assignment vector
    gamma : float
        Resolution parameter used in original modularity maximization
    n_perm : int, optional
        Number of permutations to test against. Default: 10000
    alpha : (0,1) float, optional
        Alpha level to assess signifiance. Default: 0.01
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    ndarray
        Significance of each community in `comm` (boolean)

    See Also
    --------
    netneurotools.modularity.get_modularity_z
    netneurotools.modularity.get_modularity_sig
    """

    rs = check_random_state(seed)

    real_qs = get_modularity(adjacency, comm, gamma)
    simu_qs = np.empty(shape=(np.unique(comm).size, n_perm))
    for perm in range(n_perm):
        simu_qs[:, perm] = get_modularity(adjacency,
                                          rs.permutation(comm),
                                          gamma)

    q_sig = real_qs > np.percentile(simu_qs, 100 * (1 - alpha), axis=1)

    return q_sig
