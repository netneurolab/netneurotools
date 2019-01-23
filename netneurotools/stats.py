# -*- coding: utf-8 -*-
"""
Functions for performing statistical preprocessing and analyses
"""

import numpy as np
from scipy.stats import zmap

from . import utils


def residualize(X, Y, Xc=None, Yc=None, normalize=True, add_intercept=True):
    """
    Residualizes `X` from `Y`, optionally using betas from `Yc ~ Xc`

    Parameters
    ----------
    X : (S1[, R]) array_like
        Coefficient matrix of `R` variables for `S1` subjects
    Y : (S1[, F]) array_like
        Dependent variable matrix of `F` variables for `S1` subjects
    Xc : (S2[, R]) array_like, optional
        Coefficient matrix of `R` variables for `S2` subjects. If not specified
        then `X` is used to estimate betas. Default: None
    Yc : (S2[, F]) array_like, optional
        Dependent variable matrix of `F` variables for `S1` subjects. If not
        specified then `Y` is used to estimate betas. Default: None
    normalize : bool, optional
        Whether to normalize (i.e., z-score) residuals. Default: True
    add_intercept : bool, optional
        Whether to add intercept to `X` (and `Xc`, if provided). The intercept
        will not be included in the residualizing process, just the beta
        estimation. Default: True

    Returns
    -------
    Y_resid : (N, F) numpy.ndarray
        Residuals of `Y ~ X`
    """

    if ((Yc is None and Xc is not None) or (Yc is not None and Xc is None)):
        raise ValueError('If processing against a comparative group, you must '
                         'provide both `comp` and `comp_reg` variables.')

    if Yc is None:
        Yc, Xc = Y.copy(), X.copy()

    # add intercept to regressors if requested and calculate fit
    if add_intercept:
        X, Xc = utils.add_constant(X), utils.add_constant(Xc)
    betas, *rest = np.linalg.lstsq(Xc, Yc, rcond=None)

    # remove intercept from regressors and betas for calculation of residuals
    if add_intercept:
        betas = betas[:-1]
        X, Xc = X[:, :-1], Xc[:, :-1]

    # calculate residuals
    Y_resid = Y - (X @ betas)
    Yc_resid = Yc - (Xc @ betas)

    if normalize:
        Y_resid = zmap(Y_resid, compare=Yc_resid)

    return Y_resid


def get_mad_outliers(data, thresh=3.5):
    """
    Determines which samples in `data` are outliers

    Uses the Median Absolute Deviation for determining whether datapoints are
    outliers

    Parameters
    ----------
    data : (N, M) array_like
        Data array where `N` is samples and `M` is features
    thresh : float, optional
        Modified z-score. Observations with a modified z-score (based on the
        median absolute deviation) greater than this value will be classified
        as outliers. Default: 3.5

    Returns
    -------
    outliers : (N,) numpy.ndarray
        Boolean array where True indicates an outlier

    References
    ----------
    Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
    Handle Outliers", The ASQC Basic References in Quality Control: Statistical
    Techniques, Edward F. Mykytka, Ph.D., Editor.

    Notes
    -----
    Taken directly from https://stackoverflow.com/a/22357811.
    """

    data = np.asarray(data)

    if data.ndim == 1:
        data = np.vstack(data)
    if data.ndim > 2:
        data = data.reshape(len(data), -1)

    median = np.nanmedian(data, axis=0)
    diff = np.nansum((data - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
