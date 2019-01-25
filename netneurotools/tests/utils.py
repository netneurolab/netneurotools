# -*- coding: utf-8 -*-
"""
Utility functions for generating test data
"""

import numpy as np
from sklearn.utils.validation import check_random_state


def make_correlated_xy(corr=0.85, size=10000, seed=None):
    """
    Generates two random vectors that are correlated to approximately `corr`

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

    Returns
    -------
    vectors : numpy.ndarray
        Random vectors of size ``size`` with correlation specified by `corr`
    """

    rs = check_random_state(seed)

    # if we're given a single number, assume two vectors are desired
    if isinstance(corr, (int, float)):
        covs = np.ones((2, 2)) * 0.111
        covs[(0, 1), (1, 0)] *= corr
    # if we're given a correlation matrix, assume `N` vectors are desired
    elif isinstance(corr, (list, np.ndarray)):
        corr = np.asarray(corr)
        covs = corr * 0.111
    means = [0] * len(covs)

    # generate the variables
    vectors = rs.multivariate_normal(mean=means, cov=covs, size=size).T

    return vectors
