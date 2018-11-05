# -*- coding: utf-8 -*-
"""
Functions for working with network modularity
"""

import bct
import numpy as np
from sklearn.utils.validation import check_random_state
from . import algorithms


def consensus_modularity(adjacency, gamma=1, B='modularity',
                         repeats=250, null_func=np.mean):
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
    null_func : function, optional
        Null function for generation of resolution parameter for reclustering.
        Default: `np.mean`

    Returns
    -------
    consensus : (N,) np.ndarray
        Consensus-derived community assignments
    Q_all : array_like
        Optimized modularity over all `repeats` community assignments
    zrand_all : float
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

    comms = np.column_stack(comms)  # stack community partitions

    # generate probability matrix of node co-assignment
    ag = bct.clustering.agreement(comms, buffsz=comms.shape[0]) / repeats

    # generate null probability matrix via permutation
    comms_null = np.zeros_like(comms)
    for n, i in enumerate(comms.T):
        comms_null[:, n] = np.random.permutation(i)
    ag_null = bct.clustering.agreement(comms_null, buffsz=len(comms)) / repeats

    # consensus cluster the original probability matrix with null as threshold
    consensus = bct.clustering.consensus_und(ag, null_func(ag_null), 10)

    # get z-rand statistics for partition similarity (n.b. can take a while)
    zrand_all = algorithms.zrand_partitions(comms)

    return consensus, Q_all, zrand_all


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
    seed : int, optional
        Seed for reproducibility. Default: None

    Returns
    -------
    q_z : float
        Average Z-score of modularity of communities
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
    seed : int, optional
        Seed for reproducibility. Default: None

    Returns
    -------
    ndarray
        Significance of each community in `comm` (boolean)
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
