#!/usr/bin/env python

import numpy as np


def ecdf(data):
    """
    Estimates empirical cumulative distribution function of `data`

    Parameters
    ----------
    data : array_like
        Array containing values from a continuous distribution

    Returns
    -------
    cumprob : ndarray
        Cumulative probability of distribution
    quantiles : ndarray
        Quantiles of distribution

    Notes
    -----
    Taken from StackOverflow: https://stackoverflow.com/a/33346366.
    """

    sample = np.atleast_1d(data)
    quantiles, counts = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype('float') / sample.size

    return cumprob, quantiles


def fcn_group_average(data, n_boot=1000, alpha=0.05, seed=None):
    """
    Calculates group-level, thresholded functional connectivity matrix

    This function concatenates all the subject time series and computes a
    correlation matrix based on this extended time series. It then generates
    bootstrapped samples from the concatenated matrix (of length `t`) and
    estimates confidence intervals around the correlations. Correlations whose
    sign is consistent across bootstraps are retained; others are set to 0
    (i.e., the correlation matrix is thresholded).

    Parameters
    ----------
    data : (N x T x S) array_like
        Pre-processed functional time series of shape, where `N` is the number
        of nodes, `T` is the number of volumes in the time series, and `S` is
        the number of subjects
    n_boot : int, optional
        Number of bootstraps to generate correlation CIs. Default: 1000
    alpha : float, optional
        Alpha to assess CIs, within (0,1). Default: 0.05
    seed : int, optional
        Random seed. Default: None

    Returns
    -------
    consensus : (N x N) ndarray
        Thresholded, group-level correlation matrix
    """

    if seed is not None: np.random.seed(seed)

    if alpha > 1 or alpha < 0:
        raise ValueError("`alpha` must be between 0 and 1.")

    collapsed_data = data.reshape((len(data),-1), order='F')
    consensus = np.corrcoef(collapsed_data)

    bootstrapped_corrmat = np.zeros((len(data), len(data), n_boot))

    # generate `n_boot` bootstrap correlation matrices by sampling `t` time
    # points from the concatenated time series
    for boot in range(n_boot):
        indices = np.random.randint(collapsed_data.shape[-1],
                                    size=data.shape[1])
        bootstrapped_corrmat[:,:,boot] = np.corrcoef(collapsed_data[:,indices])

    # extract the CIs from the bootstrapped correlation matrices
    alpha = 100*(alpha/2)
    bounds = [alpha, 100-alpha]
    bootstrapped_ci = np.percentile(bootstrapped_corrmat, bounds, axis=-1)

    # remove unreliable (i.e., CI zero-crossing) correlations
    indices_to_keep = np.sign(bootstrapped_ci).sum(axis=0).astype('bool')
    consensus[~indices_to_keep] = 0

    return consensus


def ijk_xyz_input_check(coords):
    """
    Confirms inputs to `ijk_to_xyz()` and `xyz_to_ijk()` are in proper format

    Parameters
    ----------
    coords : array_like
        Coordinates to be transformed

    Returns
    -------
    coords : (3 x N) ndarray
        Coordinates to be transformed
    """

    coords = np.atleast_2d(coords)
    if coords.shape[0] != 3:
        coords = coords.T
    if coords.shape[0] != 3:
        raise ValueError("Input coordinates must be of shape (3,N).")

    return coords


def ijk_to_xyz(coordinates, affine):
    """
    Converts voxel `coordinates` in cartesian space to `affine` space

    Parameters
    ----------
    coordinates : (3 x N) array_like
        i, j, k values of coordinates to be transformed
    affine : (3 x 4) array_like
        Affine matrix containing displacement + boundary information

    Returns
    -------
    (3 x N) ndarray
        Provided `coordinates` in `affine` space
    """

    coordinates = ijk_xyz_input_check(coordinates)

    return (affine[:,:-1] @ coordinates) + affine[:,[-1]]


def xyz_to_ijk(coordinates, affine):
    """
    Converts voxel `coordinates` in `affine` space to cartesian space

    Parameters
    ----------
    coordinates : (3 X N) array_like
        x, y, z values of coordinates to be transformed
    affine : (3 X 4) array_like
        Affine matrix containing displacement + boundary information

    Returns
    -------
    (3 X N) ndarray
        Provided `coordinates` in cartesian space
    """

    coordinates = ijk_xyz_input_check(coordinates)

    trans = coordinates - affine[:,[-1]]

    return np.linalg.solve(affine[:,:-1], trans)
