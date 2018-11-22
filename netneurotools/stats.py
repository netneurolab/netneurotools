# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.validation import check_array

from . import utils


def zscore(data, axis=0, ddof=1, comp=None):
    """
    Z-scores ``X`` by subtracting mean and dividing by standard deviation

    Effectively the same as ``np.nan_to_num(scipy.stats.zscore(X))`` but
    handles DivideByZero without issuing annoying warnings.

    Parameters
    ----------
    data : (N[, ...]) array_like
        Data to be z-scored
    axis : int, optional
        Axis on which to z-score `data`. Default: 0
    ddof : int, optional
        Delta degrees of freedom.  The divisor used in calculations is
        `M - ddof`, where `M` is the number of elements along `axis`.
        Default: 1
    comp : (M[, ...]) array_like, optional
        Distribution to use for z-scoring `data`. Should have same dimension as
        `data` along `axis`. Default: `data`

    Returns
    -------
    zstat : (N[, ...]) np.ndarray
        Z-scored version of ``data``
    """

    data = check_array(data, ensure_2d=False, allow_nd=True)

    if comp is not None:
        comp = check_array(comp, ensure_2d=False, allow_nd=True)
    else:
        comp = data

    avg = comp.mean(axis=axis, keepdims=True)
    stdev = comp.std(axis=axis, ddof=ddof, keepdims=True)
    zeros = stdev == 0

    if np.any(zeros):
        avg[zeros] = 0
        stdev[zeros] = 1

    zstat = (data - avg) / stdev
    zstat[np.repeat(zeros, zstat.shape[axis], axis=axis)] = 0

    return zstat


def regress_from_comp(base, base_reg, comp=None, comp_reg=None,
                      normalize=True, add_intercept=True):
    """
    Residualizes and optionally z-scores `base` using `comp`

    Regresses `base_reg` from `base` using betas derived from `comp ~ comp_reg
    + C` (where `C` is the intercept) and then optionally z-scores resultant
    residuals using distribution of residuals from `comp ~ comp_reg + C`.
    The intercept (`C`) is NOT regressed from `base`.

    If `comp` is not provided `base` is used to estimate betas.

    Parameters
    ----------
    base : (S1, F) array_like
        Data for `S1` subjects across `F` features
    base_reg : (S1,) array_like
        Regressors for `S1` subjects, corresponding to those in `base`
    comp : (S2, F) array_like, optional
        Data for `S2` subjects across `F` features, where `F` is the same
        features as in `base`. If not specified, `base` is used instead.
        Default: None
    comp_reg : (S2,) array_like, optional
        Regressors for `S2` subjects, corresponding to those in `comp`. If not
        specified, `base_reg` is used instead. Default: None
    normalize : bool, optional
        Whether to normalize the residualized data. Default: True
    add_intercept : bool, optional
        Whether to add intercept to `base_reg` and `comp_reg`. This will NOT
        be regressed out of the data. Default: True

    Returns
    -------
    processed : (N, F) array_like
        Residualize and optionally normalized data from `base`
    """

    if ((comp is None and comp_reg is not None)
            or (comp is not None and comp_reg is None)):
        raise ValueError('If processing against a comparative group, you must '
                         'provide both `comp` and `comp_reg` variables.')

    if comp is None:
        comp, comp_reg = base.copy(), base_reg.copy()

    # add intercept to regressors if requested and calculate fit
    if add_intercept:
        base_reg = utils.add_constant(base_reg)
        comp_reg = utils.add_constant(comp_reg)
    betas, *rest = np.linalg.lstsq(comp_reg, comp, rcond=None)

    # remove intercept from regressors and betas for calculation of residuals
    if add_intercept:
        betas = betas[:-1]
        base_reg, comp_reg = base_reg[:, :-1], comp_reg[:, :-1]

    # calculate residuals
    comp_resid = comp - (comp_reg @ betas)
    processed = base - (base_reg @ betas)

    if normalize:
        processed = zscore(processed, comp=comp_resid)

    return processed


def get_mad_outliers(data, thresh=3.5):
    """
    Determines which samples in `data` are outliers

    Uses the Median Absolute Deviation for determining whether datapoints are
    outliers

    Parameters:
    -----------
    data : (N, M) array_like
        Data array where `N` is samples and `M` is features
    thresh : float, optional
        Modified z-score. Observations with a modified z-score (based on the
        median absolute deviation) greater than this value will be classified
        as outliers. Default: 3.5

    Returns:
    --------
    outliers : (N,) numpy.ndarray
        Boolean array where True indicates an outlier

    References:
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
