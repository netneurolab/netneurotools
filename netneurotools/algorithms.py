# -*- coding: utf-8 -*-

import numpy as np
try:
    from numba import njit, prange
    use_numba = True
except ImportError:
    prange = range
    use_numba = False


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

    References
    ----------
    .. [1] `Amanda L. Traud, Eric D. Kelsic, Peter J. Mucha, and Mason A.
       Porter. (2011). Comparing Community Structure to Characteristics in
       Online Collegiate Social Networks. SIAM Review, 53, 526-543
       <https://arxiv.org/abs/0809.0690>`_
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


def zrand_partitions(communities):
    """
    Calculates average and std of z-Rand for all pairs of community assignments

    Iterates through every pair of community assignment vectors in
    `communities` and calculates the z-Rand score to assess their similarity.
    Returns the mean and standard deviation of all z-Rand scores.

    Parameters
    ----------
    communities : (S, R) array_like
        Community assignments for `S` samples over `R` partitions

    Returns
    -------
    zrand_avg : float
        Average z-Rand score over pairs of community assignments
    zrand_std : float
        Standard deviation of z-Rand over pairs of community assignments
    """

    n_partitions = communities.shape[-1]
    all_zrand = np.zeros(int(n_partitions * (n_partitions - 1) / 2))

    for c1 in prange(n_partitions):
        for c2 in prange(c1 + 1, n_partitions):
            idx = int((c1 * n_partitions) + c2 - ((c1 + 1) * (c1 + 2) // 2))
            all_zrand[idx] = zrand(communities[:, c1], communities[:, c2])

    return np.nanmean(all_zrand), np.nanstd(all_zrand)


if use_numba:
    _dummyvar = njit(_dummyvar)
    zrand = njit(zrand)
    zrand_partitions = njit(zrand_partitions, parallel=True)
