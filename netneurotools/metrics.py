# -*- coding: utf-8 -*-
"""
Functions for calculating various metrics from networks
"""

import numpy as np
from scipy.linalg import expm


def communicability(adjacency):
    """
    Computes the communicability of pairs of nodes in `adjacency`

    Parameters
    ----------
    adjacency : (N x N) array_like
        Unweighted, direct/undirected connection weight/length array

    Returns
    -------
    (N x N) ndarray
        Symmetric array representing communicability of nodes {i, j}

    See Also
    --------
    netneurotools_matlab/communicability.m

    References
    ----------
    Estrada, E., & Hatano, N. (2008). Communicability in complex networks.
    Physical Review E, 77(3), 036111.

    Examples
    --------
    >>> from netneurotools.metrics import communicability
    >>> A = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1]])
    >>> Q = communicability(A)
    >>> Q
    array([[4.19452805, 0.        , 3.19452805],
           [1.47624622, 2.71828183, 3.19452805],
           [3.19452805, 0.        , 4.19452805]])
    """

    if not np.any(np.logical_or(adjacency == 0, adjacency == 1)):
        raise ValueError('Provided adjancecy matrix must be unweighted.')

    return expm(adjacency)


def communicability_wei(adjacency):
    """
    Computes the communicability of pairs of nodes in `adjacency`

    Parameters
    ----------
    adjacency : (N x N) array_like
        Weighted, direct/undirected connection weight/length array

    Returns
    -------
    cmc : (N x N) ndarray
        Symmetric array representing communicability of nodes {i, j}

    See Also
    --------
    netneurotools_matlab/communicability_wei.m

    References
    ----------
    Estrada, E., & Hatano, N. (2008). Communicability in complex networks.
    Physical Review E, 77(3), 036111.

    Examples
    --------
    >>> from netneurotools.metrics import communicability_wei
    >>> A = np.array([[2, 0, 3], [0, 2, 1], [0.5, 0, 1]])
    >>> Q = communicability_wei(A)
    >>> Q
    array([[0.        , 0.        , 1.93581903],
           [0.07810379, 0.        , 0.94712177],
           [0.32263651, 0.        , 0.        ]])
    """

    row_sum = adjacency.sum(1)
    neg_sqrt = np.power(row_sum, -0.5)
    square_sqrt = np.diag(neg_sqrt)
    for_expm = square_sqrt @ adjacency @ square_sqrt

    cmc = expm(for_expm)
    cmc[np.diag_indices_from(cmc)] = 0

    return cmc
