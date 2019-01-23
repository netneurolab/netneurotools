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
    """

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
    """

    row_sum = adjacency.sum(1)
    neg_sqrt = np.power(row_sum, -0.5)
    square_sqrt = np.diag(neg_sqrt)
    for_expm = square_sqrt @ adjacency @ square_sqrt

    cmc = expm(for_expm)
    cmc[np.diag_indices_from(cmc)] = 0

    return cmc
