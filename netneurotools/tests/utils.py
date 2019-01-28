# -*- coding: utf-8 -*-
"""
Utility functions for generating test data
"""

import numpy as np
from sklearn.utils.validation import check_random_state


def make_correlated_xy(corr=0.85, size=10000, seed=None, tol=0.05):
    """
    Generates random vectors that are correlated to approximately `corr`

    Parameters
    ----------
    corr : [-1, 1] float or (N, N) numpy.ndarray, optional
        The approximate correlation desired. If a float is provided, two
        vectors with the specified level of correlation will be generated. If
        an array is provided, it is assumed to be a symmetrical correlation
        matrix and ``len(corr)`` vectors with the specified levels of
        correlation will be generated. Default: 0.85
    size : int or tuple, optional
        Desired size of the generated vectors. Default: 1000
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None
    tol : [0, 1] float, optional
        Tolerance of correlation between generated `vectors` and specified
        `corr`. Default: 0.05

    Returns
    -------
    vectors : numpy.ndarray
        Random vectors of size `size` with correlation specified by `corr`
    """

    rs = check_random_state(seed)

    # no correlations outside [-1, 1] bounds
    if np.any(np.abs(corr) > 1):
        raise ValueError('Provided `corr` must be in range [-1, 1].')

    # if we're given a single number, assume two vectors are desired
    if isinstance(corr, (int, float)):
        covs = np.ones((2, 2)) * 0.111
        covs[(0, 1), (1, 0)] *= corr
    # if we're given a correlation matrix, assume `N` vectors are desired
    elif isinstance(corr, (list, np.ndarray)):
        corr = np.asarray(corr)
        if corr.ndim != 2 or len(corr) != len(corr.T):
            raise ValueError('If `corr` is a list or array, must be a 2D '
                             'square array, not {}'.format(corr.shape))
        if np.any(np.diag(corr) != 1):
            raise ValueError('Diagonal of `corr` must be 1.')
        covs = corr * 0.111
    means = [0] * len(covs)

    # generate the variables
    count = 0
    while count < 500:
        vectors = rs.multivariate_normal(mean=means, cov=covs, size=size).T
        if np.any(np.abs(np.corrcoef(vectors) - (covs / 0.111)) > tol):
            break
        count += 1

    return vectors
