# -*- coding: utf-8 -*-
"""
Functions for calculating network metrics. Uses naming conventions adopted
from the Brain Connectivity Toolbox (https://sites.google.com/site/bctnet/).
"""

import numpy as np
from scipy.linalg import expm
from scipy.stats import ttest_ind
from bct import degrees_und

def communicability_bin(adjacency, normalize=False):
    """
    Computes the communicability of pairs of nodes in `adjacency`

    Parameters
    ----------
    adjacency : (N, N) array_like
        Unweighted, direct/undirected connection weight/length array
    normalize : bool, optional
        Whether to normalize `adjacency` by largest eigenvalue prior to
        calculation of communicability metric. Default: False

    Returns
    -------
    comm : (N, N) numpy.ndarray
        Symmetric array representing communicability of nodes {i, j}

    References
    ----------
    Estrada, E., & Hatano, N. (2008). Communicability in complex networks.
    Physical Review E, 77(3), 036111.

    Examples
    --------
    >>> from netneurotools import metrics

    >>> A = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1]])
    >>> Q = metrics.communicability_bin(A)
    >>> Q
    array([[4.19452805, 0.        , 3.19452805],
           [1.47624622, 2.71828183, 3.19452805],
           [3.19452805, 0.        , 4.19452805]])
    """

    if not np.any(np.logical_or(adjacency == 0, adjacency == 1)):
        raise ValueError('Provided adjancecy matrix must be unweighted.')

    # normalize by largest eigenvalue to prevent communicability metric from
    # "blowing up"
    if normalize:
        norm = np.linalg.eigvals(adjacency).max()
        adjacency = adjacency / norm

    return expm(adjacency)


def communicability_wei(adjacency):
    """
    Computes the communicability of pairs of nodes in `adjacency`

    Parameters
    ----------
    adjacency : (N, N) array_like
        Weighted, direct/undirected connection weight/length array

    Returns
    -------
    cmc : (N, N) numpy.ndarray
        Symmetric array representing communicability of nodes {i, j}

    References
    ----------
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure
    applied to complex brain networks. Journal of the Royal Society Interface,
    6(33), 411-414.

    Examples
    --------
    >>> from netneurotools import metrics

    >>> A = np.array([[2, 0, 3], [0, 2, 1], [0.5, 0, 1]])
    >>> Q = metrics.communicability_wei(A)
    >>> Q
    array([[0.        , 0.        , 1.93581903],
           [0.07810379, 0.        , 0.94712177],
           [0.32263651, 0.        , 0.        ]])
    """

    # negative square root of nodal degrees
    row_sum = adjacency.sum(1)
    neg_sqrt = np.power(row_sum, -0.5)
    square_sqrt = np.diag(neg_sqrt)

    # normalize input matrix
    for_expm = square_sqrt @ adjacency @ square_sqrt

    # calculate matrix exponential of normalized matrix
    cmc = expm(for_expm)
    cmc[np.diag_indices_from(cmc)] = 0

    return cmc

def rich_feeder_peripheral(x, sc, stat = 'median'):
    """
    Calculates median or mean "connectivity" in rich (hub to hub), feeder (hub to non-hub),
    and peripheral (non-hub to non-hub) links.
    
    This function takes a square symmetric correlation/connectivity matrix `x` 
    and computes the median or mean value within rich, feeder, and peripheral links.
    Hubs are defined againt a threshold (from 0 to the maximum degree `k`,
    defined on the structural connectivity matrix, `sc`). 
    

    Parameters
    ----------
    x : (N, N) symmetric numpy.ndarray.
    sc : (N, N) symmetric numpy.ndarray
                binary structural connectivity matrix 
    stat : 'median' (default) or 'mean'.
            statistic to use over rich/feeder/peripheral links.

    Returns
    -------
    rfp : (3, k) numpy.ndarray of median rich (0), feeder (1), and peripheral (2)
          values, defined by `x`. `k` is the maximum degree defined on `sc`.
    pvals : (3, k) numpy.ndarray p-value for each link, computed using Welch's t-test.
            Rich links are compared against non-rich links. Feeder links are
            compared against peripheral links. Peripheral links are compared
            against feeder links. T-test is two-sided.
            
    Author
    ------
    This code was written by Justine Hansen who promises to fix and even
    optimize the code should any issues arise, provided you let her know.

    """
    nnodes = len(sc)
    mask = np.triu(np.ones(nnodes),1) > 0
    node_degree = degrees_und(sc)
    k = np.max(node_degree).astype(np.int64)
    rfp_label = np.zeros((len(sc[mask]),k))
    
    for i in range(k):                        # for each degree threshold
        hub_idx = np.where(node_degree >= i)  # find the hubs
        hub = np.zeros([nnodes,1])
        hub[hub_idx,:] = 1
        
        rfp = np.zeros([nnodes,nnodes])       # for each link, define rfp
        for ii in range(nnodes):
            for iii in range(nnodes):
                if hub[ii] + hub[iii] == 2:
                    rfp[ii,iii] = 1 # rich
                if hub[ii] + hub[iii] == 1:
                    rfp[ii,iii] = 2 # feeder
                if hub[ii] + hub[iii] == 0:
                    rfp[ii,iii] = 3 # peripheral
        rfp_label[:,i] = rfp[mask]
    
    rfp = np.zeros([3,k])
    pvals = np.zeros([3,k])
    for i in range(k):

        if stat == 'median':
            rfp[0,i] = np.median(x[rfp_label[:,i] == 1]) # rich
            rfp[1,i] = np.median(x[rfp_label[:,i] == 2]) # feeder
            rfp[2,i] = np.median(x[rfp_label[:,i] == 3]) # peripheral
            
        if stat == 'mean':
            rfp[0,i] = np.mean(x[rfp_label[:,i] == 1]) # rich
            rfp[1,i] = np.mean(x[rfp_label[:,i] == 2]) # feeder
            rfp[2,i] = np.mean(x[rfp_label[:,i] == 3]) # peripheral
        
        # p-value
        _, pvals[0,i] = ttest_ind(x[rfp_label[:,i] == 1], x[rfp_label[:,i] != 1], equal_var=False) # Welch's t-test
        _, pvals[1,i] = ttest_ind(x[rfp_label[:,i] == 2], x[rfp_label[:,i] == 3], equal_var=False) 
        _, pvals[2,i] = ttest_ind(x[rfp_label[:,i] == 3], x[rfp_label[:,i] == 2], equal_var=False) 
        
    return rfp, pvals