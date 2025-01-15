"""Functions for calculating correlation."""

import numpy as np
import scipy.stats as sstats
import scipy.special as sspecial
from sklearn.utils.validation import check_random_state

try:  # scipy >= 1.8.0
    from scipy.stats._stats_py import _chk2_asarray
except ImportError:  # scipy < 1.8.0
    from scipy.stats.stats import _chk2_asarray


from .. import has_numba

if has_numba:
    from numba import njit


def efficient_pearsonr(a, b, ddof=1, nan_policy="propagate"):
    """
    Compute correlation of matching columns in `a` and `b`.

    Parameters
    ----------
    a,b : array_like
        Sample observations. These arrays must have the same length and either
        an equivalent number of columns or be broadcastable
    ddof : int, optional
        Degrees of freedom correction in the calculation of the standard
        deviation. Default: 1
    nan_policy : bool, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default: 'propagate'

    Returns
    -------
    corr : float or numpy.ndarray
        Pearson's correlation coefficient between matching columns of inputs
    pval : float or numpy.ndarray
        Two-tailed p-values

    Notes
    -----
    If either input contains nan and nan_policy is set to 'omit', both arrays
    will be masked to omit the nan entries.

    Examples
    --------
    >>> from netneurotools import stats

    Generate some not-very-correlated and some highly-correlated data:

    >>> np.random.seed(12345678)  # set random seed for reproducible results
    >>> x1, y1 = stats.make_correlated_xy(corr=0.1, size=100)
    >>> x2, y2 = stats.make_correlated_xy(corr=0.8, size=100)

    Calculate both correlations simultaneously:

    >>> stats.efficient_pearsonr(np.c_[x1, x2], np.c_[y1, y2])
    (array([0.10032565, 0.79961189]), array([3.20636135e-01, 1.97429944e-23]))
    """
    a, b, _ = _chk2_asarray(a, b, 0)
    if len(a) != len(b):
        raise ValueError("Provided arrays do not have same length")

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    if nan_policy not in ("propagate", "raise", "omit"):
        raise ValueError(f'Value for nan_policy "{nan_policy}" not allowed')

    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    if a.shape[1] != b.shape[1]:
        a, b = np.broadcast_arrays(a, b)

    mask = np.logical_or(np.isnan(a), np.isnan(b))
    if nan_policy == "raise" and np.any(mask):
        raise ValueError('Input cannot contain NaN when nan_policy is "omit"')
    elif nan_policy == "omit":
        # avoid making copies of the data, if possible
        a = np.ma.masked_array(a, mask, copy=False, fill_value=np.nan)
        b = np.ma.masked_array(b, mask, copy=False, fill_value=np.nan)

    with np.errstate(invalid="ignore"):
        corr = sstats.zscore(a, ddof=ddof, nan_policy=nan_policy) * sstats.zscore(
            b, ddof=ddof, nan_policy=nan_policy
        )

    sumfunc, n_obs = np.sum, len(a)
    if nan_policy == "omit":
        corr = corr.filled(np.nan)
        sumfunc = np.nansum
        n_obs = np.squeeze(np.sum(np.logical_not(np.isnan(corr)), axis=0))

    corr = sumfunc(corr, axis=0) / (n_obs - 1)
    corr = np.squeeze(np.clip(corr, -1, 1)) / 1

    # taken from scipy.stats
    ab = (n_obs / 2) - 1
    prob = 2 * sspecial.betainc(ab, ab, 0.5 * (1 - np.abs(corr)))

    return corr, prob


def fast_pearsonr():
    """Calculate Pearson correlation coefficient."""
    pass


def _weighted_mean(x, w):
    return np.sum(x * w) / np.sum(w)


if has_numba:
    _weighted_mean = njit(_weighted_mean)


def _weighted_pearsonr_vectorized(x_vec, y_vec, weight_vec):
    x_bar_diff = x_vec - _weighted_mean(x_vec, weight_vec)
    y_bar_diff = y_vec - _weighted_mean(y_vec, weight_vec)
    weight_vec_sum = np.sum(weight_vec)
    cov_x_y = np.sum(weight_vec * x_bar_diff * y_bar_diff) / weight_vec_sum
    cov_x_x = np.sum(weight_vec * x_bar_diff * x_bar_diff) / weight_vec_sum
    cov_y_y = np.sum(weight_vec * y_bar_diff * y_bar_diff) / weight_vec_sum
    return cov_x_y / np.sqrt(cov_x_x * cov_y_y)


def _weighted_pearsonr_numba(x_vec, y_vec, weight_vec):
    n = len(x_vec)
    x_weighted_mean = _weighted_mean(x_vec, weight_vec)
    y_weighted_mean = _weighted_mean(y_vec, weight_vec)
    upper, lower1, lower2 = 0, 0, 0
    for i in range(n):
        upper += (
            weight_vec[i] * (x_vec[i] - x_weighted_mean) * (y_vec[i] - y_weighted_mean)
        )
        lower1 += weight_vec[i] * (x_vec[i] - x_weighted_mean) ** 2
        lower2 += weight_vec[i] * (y_vec[i] - y_weighted_mean) ** 2
    return upper / np.sqrt(lower1 * lower2)


if has_numba:
    _weighted_pearsonr_numba = njit(_weighted_pearsonr_numba)


def weighted_pearsonr(x_vec, y_vec, weight_vec, use_numba=has_numba):
    r"""
    Calculate weighted Pearson correlation coefficient.

    Parameters
    ----------
    x_vec : array_like
        First vector of data
    y_vec : array_like
        Second vector of data
    weight_vec : array_like
        Vector of weights
    use_numba : bool, optional
        Whether to use numba for calculation. Default: True
        (if numba is available).

    Returns
    -------
    corr : float
        Weighted Pearson correlation coefficient

    Notes
    -----
    This function calculates the weighted Pearson correlation coefficient between
    two vectors, defined as:

    .. math::
        r = \frac{\sum_i w_i (x_i - \bar{x})(y_i - \bar{y})}
                    {\sqrt{\sum_i w_i (x_i - \bar{x})^2 \sum_i w_i (y_i - \bar{y})^2}}

    where :math:`x_i` and :math:`y_i` are the data points, :math:`w_i` are the
    weights, and :math:`\bar{x}` and :math:`\bar{y}` are the weighted means of
    the data points.

    """
    if use_numba:
        if not has_numba:
            raise ValueError("Numba not installed; cannot use numba for calculation")
        return _weighted_pearsonr_numba(x_vec, y_vec, weight_vec)
    else:
        return _weighted_pearsonr_vectorized(x_vec, y_vec, weight_vec)


def make_correlated_xy(corr=0.85, size=10000, seed=None, tol=0.001):
    """
    Generate random vectors that are correlated to approximately `corr`.

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
        `corr`. Default: 0.001

    Returns
    -------
    vectors : numpy.ndarray
        Random vectors of size `size` with correlation specified by `corr`

    Examples
    --------
    >>> from netneurotools import stats

    By default two vectors are generated with specified correlation

    >>> x, y = stats.make_correlated_xy()
    >>> np.corrcoef(x, y)  # doctest: +SKIP
    array([[1.        , 0.85083661],
           [0.85083661, 1.        ]])
    >>> x, y = stats.make_correlated_xy(corr=0.2)
    >>> np.corrcoef(x, y)  # doctest: +SKIP
    array([[1.        , 0.20069953],
           [0.20069953, 1.        ]])

    You can also provide correlation matrices to generate more than two vectors
    if desired. Note that this makes it more difficult to ensure the actual
    correlations are close to the desired values:

    >>> corr = [[1, 0.5, 0.3], [0.5, 1, 0], [0.3, 0, 1]]
    >>> out = stats.make_correlated_xy(corr=corr)
    >>> out.shape
    (3, 10000)
    >>> np.corrcoef(out)  # doctest: +SKIP
    array([[1.        , 0.50965273, 0.30235686],
           [0.50965273, 1.        , 0.01089107],
           [0.30235686, 0.01089107, 1.        ]])
    """
    rs = check_random_state(seed)

    # no correlations outside [-1, 1] bounds
    if np.any(np.abs(corr) > 1):
        raise ValueError("Provided `corr` must (all) be in range [-1, 1].")

    # if we're given a single number, assume two vectors are desired
    if isinstance(corr, (int, float)):
        covs = np.ones((2, 2)) * 0.111
        covs[(0, 1), (1, 0)] *= corr
    # if we're given a correlation matrix, assume `N` vectors are desired
    elif isinstance(corr, (list, np.ndarray)):
        corr = np.asarray(corr)
        if corr.ndim != 2 or len(corr) != len(corr.T):
            raise ValueError(
                "If `corr` is a list or array, must be a 2D "
                "square array, not {}".format(corr.shape)
            )
        if np.any(np.diag(corr) != 1):
            raise ValueError("Diagonal of `corr` must be 1.")
        covs = corr * 0.111
    means = [0] * len(covs)

    # generate the variables
    count = 0
    while count < 500:
        vectors = rs.multivariate_normal(mean=means, cov=covs, size=size).T
        flat = vectors.reshape(len(vectors), -1)
        # if diff between actual and desired correlations less than tol, break
        if np.all(np.abs(np.corrcoef(flat) - (covs / 0.111)) < tol):
            break
        count += 1

    return vectors
