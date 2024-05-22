"""Functions for calculating permutation test."""

import numpy as np
from sklearn.utils.validation import check_random_state

try:  # scipy >= 1.8.0
    from scipy.stats._stats_py import _chk2_asarray
except ImportError:  # scipy < 1.8.0
    from scipy.stats.stats import _chk2_asarray

from .correlation import efficient_pearsonr


def permtest_1samp(a, popmean, axis=0, n_perm=1000, seed=0):
    """
    Non-parametric equivalent of :py:func:`scipy.stats.ttest_1samp`.

    Generates two-tailed p-value for hypothesis of whether `a` differs from
    `popmean` using permutation tests

    Parameters
    ----------
    a : array_like
        Sample observations
    popmean : float or array_like
        Expected valued in null hypothesis. If array_like then it must have the
        same shape as `a` excluding the `axis` dimension
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole array
        of `a`. Default: 0
    n_perm : int, optional
        Number of permutations to assess. Unless `a` is very small along `axis`
        this will approximate a randomization test via Monte Carlo simulations.
        Default: 1000
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Set to None for "randomness".
        Default: 0

    Returns
    -------
    stat : float or numpy.ndarray
        Difference from `popmean`
    pvalue : float or numpy.ndarray
        Non-parametric p-value

    Notes
    -----
    Providing multiple values to `popmean` to run *independent* tests in
    parallel is not currently supported.

    The lowest p-value that can be returned by this function is equal to 1 /
    (`n_perm` + 1).

    Examples
    --------
    >>> from netneurotools import stats
    >>> np.random.seed(7654567)  # set random seed for reproducible results
    >>> rvs = np.random.normal(loc=5, scale=10, size=(50, 2))

    Test if mean of random sample is equal to true mean, and different mean. We
    reject the null hypothesis in the second case and don't reject it in the
    first case.

    >>> stats.permtest_1samp(rvs, 5.0)
    (array([-0.985602  , -0.05204969]), array([0.48551449, 0.95904096]))
    >>> stats.permtest_1samp(rvs, 0.0)
    (array([4.014398  , 4.94795031]), array([0.00699301, 0.000999  ]))

    Example using axis and non-scalar dimension for population mean

    >>> stats.permtest_1samp(rvs, [5.0, 0.0])
    (array([-0.985602  ,  4.94795031]), array([0.48551449, 0.000999  ]))
    >>> stats.permtest_1samp(rvs.T, [5.0, 0.0], axis=1)
    (array([-0.985602  ,  4.94795031]), array([0.51548452, 0.000999  ]))
    """
    a, popmean, axis = _chk2_asarray(a, popmean, axis)
    rs = check_random_state(seed)

    if a.size == 0:
        return np.nan, np.nan

    # ensure popmean will broadcast to `a` correctly
    if popmean.ndim != a.ndim:
        popmean = np.expand_dims(popmean, axis=axis)

    # center `a` around `popmean` and calculate original mean
    zeroed = a - popmean
    true_mean = zeroed.mean(axis=axis) / 1
    abs_mean = np.abs(true_mean)

    # this for loop is not _the fastest_ but is memory efficient
    # the broadcasting alt. would mean storing zeroed.size * n_perm in memory
    permutations = np.ones(true_mean.shape)
    for _ in range(n_perm):
        flipped = zeroed * rs.choice([-1, 1], size=zeroed.shape)  # sign flip
        permutations += np.abs(flipped.mean(axis=axis)) >= abs_mean

    pvals = permutations / (n_perm + 1)  # + 1 in denom accounts for true_mean

    return true_mean, pvals


def permtest_rel(a, b, axis=0, n_perm=1000, seed=0):
    """
    Non-parametric equivalent of :py:func:`scipy.stats.ttest_rel`.

    Generates two-tailed p-value for hypothesis of whether related samples `a`
    and `b` differ using permutation tests

    Parameters
    ----------
    a, b : array_like
        Sample observations. These arrays must have the same shape.
    axis : int or None, optional
        Axis along which to compute test. If None, compute over whole arrays
        of `a` and `b`. Default: 0
    n_perm : int, optional
        Number of permutations to assess. Unless `a` and `b` are very small
        along `axis` this will approximate a randomization test via Monte
        Carlo simulations. Default: 1000
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Set to None for "randomness".
        Default: 0

    Returns
    -------
    stat : float or numpy.ndarray
        Average difference between `a` and `b`
    pvalue : float or numpy.ndarray
        Non-parametric p-value

    Notes
    -----
    The lowest p-value that can be returned by this function is equal to 1 /
    (`n_perm` + 1).

    Examples
    --------
    >>> from netneurotools import stats

    >>> np.random.seed(12345678)  # set random seed for reproducible results
    >>> rvs1 = np.random.normal(loc=5, scale=10, size=500)
    >>> rvs2 = (np.random.normal(loc=5, scale=10, size=500)
    ...         + np.random.normal(scale=0.2, size=500))
    >>> stats.permtest_rel(rvs1, rvs2)  # doctest: +SKIP
    (-0.16506275161572695, 0.8021978021978022)

    >>> rvs3 = (np.random.normal(loc=8, scale=10, size=500)
    ...         + np.random.normal(scale=0.2, size=500))
    >>> stats.permtest_rel(rvs1, rvs3)  # doctest: +SKIP
    (2.40533726097883, 0.000999000999000999)
    """
    a, b, axis = _chk2_asarray(a, b, axis)
    rs = check_random_state(seed)

    if a.shape[axis] != b.shape[axis]:
        raise ValueError('Provided arrays do not have same length along axis')

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    # calculate original difference in means
    ab = np.stack([a, b], axis=0)
    if ab.ndim < 3:
        ab = np.expand_dims(ab, axis=-1)
    true_diff = np.squeeze(np.diff(ab, axis=0)).mean(axis=axis) / 1
    abs_true = np.abs(true_diff)

    # idx array
    reidx = np.meshgrid(*[range(f) for f in ab.shape], indexing='ij')

    permutations = np.ones(true_diff.shape)
    for _ in range(n_perm):
        # use this to re-index (i.e., swap along) the first axis of `ab`
        swap = rs.random_sample(ab.shape[:-1]).argsort(axis=axis)
        reidx[0] = np.repeat(swap[..., np.newaxis], ab.shape[-1], axis=-1)
        # recompute difference between `a` and `b` (i.e., first axis of `ab`)
        pdiff = np.squeeze(np.diff(ab[tuple(reidx)], axis=0)).mean(axis=axis)
        permutations += np.abs(pdiff) >= abs_true

    pvals = permutations / (n_perm + 1)  # + 1 in denom accounts for true_diff

    return true_diff, pvals


def permtest_pearsonr(a, b, axis=0, n_perm=1000, resamples=None, seed=0):
    """
    Non-parametric equivalent of :py:func:`scipy.stats.pearsonr`.

    Generates two-tailed p-value for hypothesis of whether samples `a` and `b`
    are correlated using permutation tests

    Parameters
    ----------
    a,b : (N[, M]) array_like
        Sample observations. These arrays must have the same length and either
        an equivalent number of columns or be broadcastable
    axis : int or None, optional
        Axis along which to compute test. If None, compute over whole arrays
        of `a` and `b`. Default: 0
    n_perm : int, optional
        Number of permutations to assess. Unless `a` and `b` are very small
        along `axis` this will approximate a randomization test via Monte
        Carlo simulations. Default: 1000
    resamples : (N, P) array_like, optional
        Resampling array used to shuffle `a` when generating null distribution
        of correlations. This array must have the same length as `a` and `b`
        and should have at least the same number of columns as `n_perm` (if it
        has more then only `n_perm` columns will be used. When not specified a
        standard permutation is used to shuffle `a`. Default: None
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Set to None for "randomness".
        Default: 0

    Returns
    -------
    corr : float or numpyndarray
        Correlations
    pvalue : float or numpy.ndarray
        Non-parametric p-value

    Notes
    -----
    The lowest p-value that can be returned by this function is equal to 1 /
    (`n_perm` + 1).

    Examples
    --------
    >>> from netneurotools import stats

    >>> np.random.seed(12345678)  # set random seed for reproducible results
    >>> x, y = stats.make_correlated_xy(corr=0.1, size=100)
    >>> stats.permtest_pearsonr(x, y)  # doctest: +SKIP
    (0.10032564626876286, 0.3046953046953047)

    >>> x, y = stats.make_correlated_xy(corr=0.5, size=100)
    >>> stats.permtest_pearsonr(x, y)  # doctest: +SKIP
    (0.500040365781984, 0.000999000999000999)

    Also works with multiple columns by either broadcasting the smaller array
    to the larger:

    >>> z = x + np.random.normal(loc=1, size=100)
    >>> stats.permtest_pearsonr(x, np.column_stack([y, z]))
    (array([0.50004037, 0.25843187]), array([0.000999  , 0.01098901]))

    or by using matching columns in the two arrays (e.g., `x` and `y` vs
    `a` and `b`):

    >>> a, b = stats.make_correlated_xy(corr=0.9, size=100)
    >>> stats.permtest_pearsonr(np.column_stack([x, a]), np.column_stack([y, b]))
    (array([0.50004037, 0.89927523]), array([0.000999, 0.000999]))
    """  # noqa
    a, b, axis = _chk2_asarray(a, b, axis)
    rs = check_random_state(seed)

    if len(a) != len(b):
        raise ValueError('Provided arrays do not have same length')

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    if resamples is not None:
        if n_perm > resamples.shape[-1]:
            raise ValueError('Number of permutations requested exceeds size '
                             'of resampling array.')

    # divide by one forces coercion to float if ndim = 0
    true_corr = efficient_pearsonr(a, b)[0] / 1
    abs_true = np.abs(true_corr)

    permutations = np.ones(true_corr.shape)
    for perm in range(n_perm):
        # permute `a` and determine whether correlations exceed original
        if resamples is None:
            ap = a[rs.permutation(len(a))]
        else:
            ap = a[resamples[:, perm]]
        permutations += np.abs(efficient_pearsonr(ap, b)[0]) >= abs_true

    pvals = permutations / (n_perm + 1)  # + 1 in denom accounts for true_corr

    return true_corr, pvals
