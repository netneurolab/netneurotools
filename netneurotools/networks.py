#!/usr/bin/env python

import itertools
import bct
import numpy as np


def zrand(X, Y):
    """
    Calculates the z-Rand index of two clustering assignments

    Parameters
    ----------
    X, Y : (N x 1) array_like
        Clustering assignment vectors

    Returns
    -------
    z_rand : float
        z-Rand index

    References
    ----------
    .. [1] Traud, A. L., Kelsic, E. D., Mucha, P. J., & Porter, M. A. (2011).
       Comparing community structure to characteristics in online collegiate
       social networks. SIAM review, 53(3), 526-543.
    """

    # we need 2d arrays for this to work; shape (n,1)
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    if X.shape[0] < X.shape[1]: X = X.T
    if Y.shape[0] < Y.shape[1]: Y = Y.T

    n = X.shape[0]

    indx = bct.utils.dummyvar(X)
    indy = bct.utils.dummyvar(Y)

    Xa = indx @ indx.T
    Ya = indy @ indy.T

    M = n*(n - 1)/2
    M1 = Xa.nonzero()[0].size/2
    M2 = Ya.nonzero()[0].size/2

    wab = np.logical_and(Xa, Ya).nonzero()[0].size/2
    muab = M1*M2/M

    nx = indx.sum(0)
    ny = indy.sum(0)

    C1 = n*(n**2 - 3*n - 2) - 8*(n + 1)*M1 + 4*sum(np.power(nx,3))
    C2 = n*(n**2 - 3*n - 2) - 8*(n + 1)*M2 + 4*sum(np.power(ny,3))

    a = M/16
    b = ((4*M1 - 2*M)**2)*((4*M2 - 2*M)**2)/(256*(M**2))
    c = C1*C2/(16*n*(n - 1)*(n - 2))
    d = ((((4*M1 - 2*M)**2) - 4*C1 - 4*M)*(((4*M2 - 2*M)**2) - 4*C2 - 4*M) /
         (64*n*(n - 1)*(n - 2)*(n - 3)))

    sigw2 = a - b + c + d
    if sigw2 < 0: return 0  # catch any negatives
    sigw = np.sqrt(sigw2)
    wz = (wab - muab)/sigw

    return wz


def zrand_partitions(communities):
    """
    Calculates average and std of z-Rand for all pairs of `communities`

    Iterates through every possible pair of community assignment vectors in
    `communities` and calculates the z-Rand score to assess their similarity.
    Returns the mean and standard deviation of all z-Rand scores.

    Parameters
    ----------
    communities : (N x R) array_like
        Community assignments of `N` nodes for `R` repeats of modularity
        maximization

    Returns
    -------
    zrand_avg : float
        Average z-Rand score over all pairs of community assignments
    zrand_std : float
        Standard deviation of z-Rand over all pairs of community assignments
    """

    all_zrand  = [zrand(f[0],f[1]) for f in
                  itertools.combinations(communities.T,2)]

    zrand_avg, zrand_std = np.nanmean(all_zrand), np.nanstd(all_zrand)

    return zrand_avg, zrand_std


def consensus_modularity(adjacency,
                         gamma=1, B='modularity',
                         repeats=250,
                         null_func=np.mean):
    """
    Parameters
    ----------
    adjacency : (N x N) array_like
        Non-negative adjacency matrix
    gamma : float, optional
        Weighting parameters used in modularity maximization. Default: 1
    B : str or array_like, optional
        Null model for modularity maximization. Default: 'modularity'
    repeats : int, optional
        Number of times to repeat community detection (via modularity
        maximization). Generated community assignments will be combined into a
        consensus matrix. Default: 250
    null_func : function, optional
        Function that can accept an array and return a single number. This is
        used during the procedure that generates the consensus community
        assignment vector from the `repeats` individual community assignment
        vectors. Default: numpy.mean

    Returns
    -------
    consensus : ndarray
        Consensus community assignments
    Q_mean : float
        Average modularity of generated community assignment partitions
    zrand_avg : float
        Average z-Rand of generated community assignment partitions
    zrand_std : float
        Standard deviation z-Rand of generated community assignment partitions

    References
    ----------
    .. [1] Bassett, D. S., Porter, M. A., Wymbs, N. F., Grafton, S. T.,
       Carlson, J. M., & Mucha, P. J. (2013). Robust detection of dynamic
       community structure in networks. Chaos: An Interdisciplinary Journal of
       Nonlinear Science, 23(1), 013142.
    """

    # generate community partitions `repeat` times
    partitions = [bct.community_louvain(adjacency,
                                        gamma=gamma,
                                        B=B) for i in range(repeats)]

    # get community labels and Qs
    comms  = np.column_stack([f[0] for f in partitions]),
    Q_mean = np.mean([f[1] for f in partitions])

    ag = bct.clustering.agreement(comms) / repeats

    # generate null agreement matrix
    comms_null = np.zeros_like(comms)
    for n, i in enumerate(comms.T): comms_null[:,n] = np.random.permutation(i)
    ag_null = bct.clustering.agreement(comms_null) / repeats

    # get `null_func` of null agreement matrix
    tau = null_func(ag_null)

    # consensus cluster the agreement matrix unsing `tau` as threshold
    consensus = bct.clustering.consensus_und(ag, tau, 10)

    # get zrand statistics for partition similarity
    zrand_avg, zrand_std = zrand_partitions(comms)

    return consensus, Q_mean, zrand_avg, zrand_std
