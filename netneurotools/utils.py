#!/usr/bin/env python

import numpy as np


def ecdf(data):
    """
    Estimates empirical cumulative distribution function of `data`

    Taken directly from StackOverflow (where else?); see original answer at
    https://stackoverflow.com/a/33346366.

    Parameters
    ----------
    data : array
        Array of continuous distribution

    Returns
    -------
    array
        Cumulative probability of distribution
    array
        Quantiles of distribution
    """

    sample = np.atleast_1d(data)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype('float') / sample.size

    return cumprob, quantiles


def fcn_group_average(data, num_boot=1000, alpha=0.05, seed=None):
    """
    Calculates group-level, thresholded functional connectivity matrix

    Takes as input a stack of subject-level time series matrices with
    dimensions `(n x t x subjects)` where `n` is the number of nodes, `t` is
    the number of volumes in the time series, and subject is the number of
    matrices in the stack.

    This function concatenates all the subject time series and computes a
    correlation matrix based on this extended time series. It then generates
    bootstrapped samples from the concatenated matrix (of length `t`) and
    estimates confidence intervals around the correlations. Correlations whose
    sign is consistent across bootstraps are retained; others are set to 0
    (i.e., the correlation matrix is thresholded).

    Parameters
    ----------
    data : array
        Pre-processed functional time series of shape `(n x t x subjects)`
        where `n` is the number of nodes and `t` is the number of volumes in
        the time series
    num_boot : int, optional
        Number of bootstraps to generate correlation CIs. Default: 1000
    alpha : float, optional
        Alpha to assess CIs, within (0,1). Default: 0.05
    seed : int, optional
         Random seed. Default: None

    Returns
    -------
    array
        Thresholded correlation matrix of size `(n x n)`
    """

    if seed is not None: np.random.seed(seed)

    # first, collapse across subjects (concatenate time series)
    collapsed_data = data.reshape((len(data),-1), order='F')

    # then, generate a correlation array from that
    group_consensus = np.corrcoef(collapsed_data)

    # make an empty array to hold the bootstraps
    bootstrapped_corrmat = np.zeros((len(data), len(data), num_boot))

    # next, let's generate 1000 bootstrap correlation matrices by picking `T`
    # samples from the concatenated time series and correlating the resultant
    # array, where `T` is the number of samples in the original time series
    # for a single subject
    for boot in range(num_boot):
        indices = np.random.randint(collapsed_data.shape[-1],
                                    size=data.shape[1])
        bootstrapped_corrmat[:,:,boot] = np.corrcoef(collapsed_data[:,indices])

    # now, we'll see whether the CI of this distribution crosses zero
    # first, we'll generate the percentile bounds based on the provided `alpha`
    alpha = 100*(alpha/2)
    bounds = [alpha, 100-alpha]

    # then, we'll extract the CIs from the bootstrapped correlation matrices
    bootstrapped_ci = np.percentile(bootstrapped_corrmat, bounds, axis=-1)

    # we want to assess if these CIs cross zero (i.e., the correlations are
    # inconsistent and should be excluded from further analysis)
    # to do that, we'll check the signs of the confidence intervals and add
    # them together -- if they're not zero (i.e., +2 or -2) then we want to
    # keep them, so we'll convert it to a boolean array
    indices_to_keep = np.sign(bootstrapped_ci).sum(axis=0).astype('bool')

    # finally, we'll threshold the original correlation array, retaining only
    # the "consistent" correlations
    group_consensus[~indices_to_keep] = 0

    return group_consensus
