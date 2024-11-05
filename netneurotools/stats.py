# -*- coding: utf-8 -*-
"""Functions for performing statistical preprocessing and analyses."""

import warnings

import numpy as np
from tqdm import tqdm
from itertools import combinations
from scipy import optimize, spatial, special, stats as sstats
try:  # scipy >= 1.8.0
    from scipy.stats._stats_py import _chk2_asarray
except ImportError:  # scipy < 1.8.0
    from scipy.stats.stats import _chk2_asarray
from sklearn.utils.validation import check_random_state
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed


from . import utils
from .metrics import _graph_laplacian

try:
    from numba import njit
    use_numba = True
except ImportError:
    use_numba = False


def residualize(X, Y, Xc=None, Yc=None, normalize=True, add_intercept=True):
    """
    Return residuals of regression equation from `Y ~ X`.

    Parameters
    ----------
    X : (N[, R]) array_like
        Coefficient matrix of `R` variables for `N` subjects
    Y : (N[, F]) array_like
        Dependent variable matrix of `F` variables for `N` subjects
    Xc : (M[, R]) array_like, optional
        Coefficient matrix of `R` variables for `M` subjects. If not specified
        then `X` is used to estimate betas. Default: None
    Yc : (M[, F]) array_like, optional
        Dependent variable matrix of `F` variables for `M` subjects. If not
        specified then `Y` is used to estimate betas. Default: None
    normalize : bool, optional
        Whether to normalize (i.e., z-score) residuals. Will use residuals from
        `Yc ~ Xc` for generating mean and variance. Default: True
    add_intercept : bool, optional
        Whether to add intercept to `X` (and `Xc`, if provided). The intercept
        will not be removed, just used in beta estimation. Default: True

    Returns
    -------
    Yr : (N, F) numpy.ndarray
        Residuals of `Y ~ X`

    Notes
    -----
    If both `Xc` and `Yc` are provided, these are used to calculate betas which
    are then applied to `X` and `Y`.
    """
    if ((Yc is None and Xc is not None) or (Yc is not None and Xc is None)):
        raise ValueError('If processing against a comparative group, you must '
                         'provide both `Xc` and `Yc`.')

    X, Y = np.asarray(X), np.asarray(Y)

    if Yc is None:
        Xc, Yc = X.copy(), Y.copy()
    else:
        Xc, Yc = np.asarray(Xc), np.asarray(Yc)

    # add intercept to regressors if requested and calculate fit
    if add_intercept:
        X, Xc = utils.add_constant(X), utils.add_constant(Xc)
    betas, *rest = np.linalg.lstsq(Xc, Yc, rcond=None)

    # remove intercept from regressors and betas for calculation of residuals
    if add_intercept:
        betas = betas[:-1]
        X, Xc = X[:, :-1], Xc[:, :-1]

    # calculate residuals
    Yr = Y - (X @ betas)
    Ycr = Yc - (Xc @ betas)

    if normalize:
        Yr = sstats.zmap(Yr, compare=Ycr)

    return Yr


def get_mad_outliers(data, thresh=3.5):
    """
    Determine which samples in `data` are outliers.

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

    Notes
    -----
    Taken directly from https://stackoverflow.com/a/22357811

    References
    ----------
    Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
    Handle Outliers", The ASQC Basic References in Quality Control: Statistical
    Techniques, Edward F. Mykytka, Ph.D., Editor.

    Examples
    --------
    >>> from netneurotools import stats

    Create array with three samples of four features each:

    >>> X = np.array([[0, 5, 10, 15], [1, 4, 11, 16], [100, 100, 100, 100]])
    >>> X
    array([[  0,   5,  10,  15],
           [  1,   4,  11,  16],
           [100, 100, 100, 100]])

    Determine which sample(s) is outlier:

    >>> outliers = stats.get_mad_outliers(X)
    >>> outliers
    array([False, False,  True])
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
    reidx = list(np.meshgrid(*[range(f) for f in ab.shape], indexing='ij'))

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
    >>> from netneurotools import datasets, stats

    >>> np.random.seed(12345678)  # set random seed for reproducible results
    >>> x, y = datasets.make_correlated_xy(corr=0.1, size=100)
    >>> stats.permtest_pearsonr(x, y)  # doctest: +SKIP
    (0.10032564626876286, 0.3046953046953047)

    >>> x, y = datasets.make_correlated_xy(corr=0.5, size=100)
    >>> stats.permtest_pearsonr(x, y)  # doctest: +SKIP
    (0.500040365781984, 0.000999000999000999)

    Also works with multiple columns by either broadcasting the smaller array
    to the larger:

    >>> z = x + np.random.normal(loc=1, size=100)
    >>> stats.permtest_pearsonr(x, np.column_stack([y, z]))
    (array([0.50004037, 0.25843187]), array([0.000999  , 0.01098901]))

    or by using matching columns in the two arrays (e.g., `x` and `y` vs
    `a` and `b`):

    >>> a, b = datasets.make_correlated_xy(corr=0.9, size=100)
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


def efficient_pearsonr(a, b, ddof=1, nan_policy='propagate'):
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
    >>> from netneurotools import datasets, stats

    Generate some not-very-correlated and some highly-correlated data:

    >>> np.random.seed(12345678)  # set random seed for reproducible results
    >>> x1, y1 = datasets.make_correlated_xy(corr=0.1, size=100)
    >>> x2, y2 = datasets.make_correlated_xy(corr=0.8, size=100)

    Calculate both correlations simultaneously:

    >>> stats.efficient_pearsonr(np.c_[x1, x2], np.c_[y1, y2])
    (array([0.10032565, 0.79961189]), array([3.20636135e-01, 1.97429944e-23]))
    """
    a, b, axis = _chk2_asarray(a, b, 0)
    if len(a) != len(b):
        raise ValueError('Provided arrays do not have same length')

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    if nan_policy not in ('propagate', 'raise', 'omit'):
        raise ValueError(f'Value for nan_policy "{nan_policy}" not allowed')

    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    if (a.shape[1] != b.shape[1]):
        a, b = np.broadcast_arrays(a, b)

    mask = np.logical_or(np.isnan(a), np.isnan(b))
    if nan_policy == 'raise' and np.any(mask):
        raise ValueError('Input cannot contain NaN when nan_policy is "omit"')
    elif nan_policy == 'omit':
        # avoid making copies of the data, if possible
        a = np.ma.masked_array(a, mask, copy=False, fill_value=np.nan)
        b = np.ma.masked_array(b, mask, copy=False, fill_value=np.nan)

    with np.errstate(invalid='ignore'):
        corr = (sstats.zscore(a, ddof=ddof, nan_policy=nan_policy)
                * sstats.zscore(b, ddof=ddof, nan_policy=nan_policy))

    sumfunc, n_obs = np.sum, len(a)
    if nan_policy == 'omit':
        corr = corr.filled(np.nan)
        sumfunc = np.nansum
        n_obs = np.squeeze(np.sum(np.logical_not(np.isnan(corr)), axis=0))

    corr = sumfunc(corr, axis=0) / (n_obs - 1)
    corr = np.squeeze(np.clip(corr, -1, 1)) / 1

    # taken from scipy.stats
    ab = (n_obs / 2) - 1
    prob = 2 * special.btdtr(ab, ab, 0.5 * (1 - np.abs(corr)))

    return corr, prob


def _gen_rotation(seed=None):
    """
    Generate random matrix for rotating spherical coordinates.

    Parameters
    ----------
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation

    Returns
    -------
    rotate_{l,r} : (3, 3) numpy.ndarray
        Rotations for left and right hemisphere coordinates, respectively
    """
    rs = check_random_state(seed)

    # for reflecting across Y-Z plane
    reflect = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # generate rotation for left
    rotate_l, temp = np.linalg.qr(rs.normal(size=(3, 3)))
    rotate_l = rotate_l @ np.diag(np.sign(np.diag(temp)))
    if np.linalg.det(rotate_l) < 0:
        rotate_l[:, 0] = -rotate_l[:, 0]

    # reflect the left rotation across Y-Z plane
    rotate_r = reflect @ rotate_l @ reflect

    return rotate_l, rotate_r


def gen_spinsamples(coords, hemiid, n_rotate=1000, check_duplicates=True,
                    method='original', exact=False, seed=None, verbose=False,
                    return_cost=False):
    """
    Return a resampling array for `coords` obtained from rotations / spins.

    Using the method initially proposed in [ST1]_ (and later modified + updated
    based on findings in [ST2]_ and [ST3]_), this function applies random
    rotations to the user-supplied `coords` in order to generate a resampling
    array that preserves its spatial embedding. Rotations are generated for one
    hemisphere and mirrored for the other (see `hemiid` for more information).

    Due to irregular sampling of `coords` and the randomness of the rotations
    it is possible that some "rotations" may resample with replacement (i.e.,
    will not be a true permutation). The likelihood of this can be reduced by
    either increasing the sampling density of `coords` or changing the
    ``method`` parameter (see Notes for more information on the latter).

    Parameters
    ----------
    coords : (N, 3) array_like
        X, Y, Z coordinates of `N` nodes/parcels/regions/vertices defined on a
        sphere
    hemiid : (N,) array_like
        Array denoting hemisphere designation of coordinates in `coords`, where
        values should be {0, 1} denoting the different hemispheres. Rotations
        are generated for one hemisphere and mirrored across the y-axis for the
        other hemisphere.
    n_rotate : int, optional
        Number of rotations to generate. Default: 1000
    check_duplicates : bool, optional
        Whether to check for and attempt to avoid duplicate resamplings. A
        warnings will be raised if duplicates cannot be avoided. Setting to
        True may increase the runtime of this function! Default: True
    method : {'original', 'vasa', 'hungarian'}, optional
        Method by which to match non- and rotated coordinates. Specifying
        'original' will use the method described in [ST1]_. Specfying 'vasa'
        will use the method described in [ST4]_. Specfying 'hungarian' will use
        the Hungarian algorithm to minimize the global cost of reassignment
        (will dramatically increase runtime). Default: 'original'
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None
    verbose : bool, optional
        Whether to print occasional status messages. Default: False
    return_cost : bool, optional
        Whether to return cost array (specified as Euclidean distance) for each
        coordinate for each rotation Default: True

    Returns
    -------
    spinsamples : (N, `n_rotate`) numpy.ndarray
        Resampling matrix to use in permuting data based on supplied `coords`.
    cost : (N, `n_rotate`,) numpy.ndarray
        Cost (specified as Euclidean distance) of re-assigning each coordinate
        for every rotation in `spinsamples`. Only provided if `return_cost` is
        True.

    Notes
    -----
    By default, this function uses the minimum Euclidean distance between the
    original coordinates and the new, rotated coordinates to generate a
    resampling array after each spin. Unfortunately, this can (with some
    frequency) lead to multiple coordinates being re-assigned the same value:

        >>> from netneurotools import stats as nnstats
        >>> coords = [[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]]
        >>> hemi = [0, 0, 1, 1]
        >>> nnstats.gen_spinsamples(coords, hemi, n_rotate=1, seed=1,
        ...                         method='original', check_duplicates=False)
        array([[0],
               [0],
               [2],
               [3]])

    While this is reasonable in most circumstances, if you feel incredibly
    strongly about having a perfect "permutation" (i.e., all indices appear
    once and exactly once in the resampling), you can set the ``method``
    parameter to either 'vasa' or 'hungarian':

        >>> nnstats.gen_spinsamples(coords, hemi, n_rotate=1, seed=1,
        ...                         method='vasa', check_duplicates=False)
        array([[1],
               [0],
               [2],
               [3]])
        >>> nnstats.gen_spinsamples(coords, hemi, n_rotate=1, seed=1,
        ...                         method='hungarian', check_duplicates=False)
        array([[0],
               [1],
               [2],
               [3]])

    Note that setting this parameter may increase the runtime of the function
    (especially for `method='hungarian'`). Refer to [ST1]_ for information on
    why the default (i.e., ``exact`` set to False) suffices in most cases.

    For the original MATLAB implementation of this function refer to [ST5]_.

    References
    ----------
    .. [ST1] Alexander-Bloch, A., Shou, H., Liu, S., Satterthwaite, T. D.,
       Glahn, D. C., Shinohara, R. T., Vandekar, S. N., & Raznahan, A. (2018).
       On testing for spatial correspondence between maps of human brain
       structure and function. NeuroImage, 178, 540-51.

    .. [ST2] Blaser, R., & Fryzlewicz, P. (2016). Random Rotation Ensembles.
       Journal of Machine Learning Research, 17(4), 1–26.

    .. [ST3] Lefèvre, J., Pepe, A., Muscato, J., De Guio, F., Girard, N.,
       Auzias, G., & Germanaud, D. (2018). SPANOL (SPectral ANalysis of Lobes):
       A Spectral Clustering Framework for Individual and Group Parcellation of
       Cortical Surfaces in Lobes. Frontiers in Neuroscience, 12, 354.

    .. [ST4] Váša, F., Seidlitz, J., Romero-Garcia, R., Whitaker, K. J.,
       Rosenthal, G., Vértes, P. E., ... & Jones, P. B. (2018). Adolescent
       tuning of association cortex in human structural brain networks.
       Cerebral Cortex, 28(1), 281-294.

    .. [ST5] https://github.com/spin-test/spin-test
    """
    methods = ['original', 'vasa', 'hungarian']
    if method not in methods:
        raise ValueError('Provided method "{}" invalid. Must be one of {}.'
                         .format(method, methods))

    if exact:
        warnings.warn('The `exact` parameter will no longer be supported in '
                      'an upcoming release. Please use the `method` parameter '
                      'instead.', DeprecationWarning, stacklevel=3)
        if exact == 'vasa' and method == 'original':
            method = 'vasa'
        elif exact and method == 'original':
            method = 'hungarian'

    seed = check_random_state(seed)

    coords = np.asanyarray(coords)
    hemiid = np.squeeze(np.asanyarray(hemiid, dtype='int8'))

    # check supplied coordinate shape
    if coords.shape[-1] != 3 or coords.squeeze().ndim != 2:
        raise ValueError('Provided `coords` must be of shape (N, 3), not {}'
                         .format(coords.shape))

    # ensure hemisphere designation array is correct
    if hemiid.ndim != 1:
        raise ValueError('Provided `hemiid` array must be one-dimensional.')
    if len(coords) != len(hemiid):
        raise ValueError('Provided `coords` and `hemiid` must have the same '
                         'length. Provided lengths: coords = {}, hemiid = {}'
                         .format(len(coords), len(hemiid)))
    if np.max(hemiid) > 1 or np.min(hemiid) < 0:
        raise ValueError('Hemiid must have values in {0, 1} denoting left and '
                         'right hemisphere coordinates, respectively. '
                         + 'Provided array contains values: {}'
                         .format(np.unique(hemiid)))

    # empty array to store resampling indices
    spinsamples = np.zeros((len(coords), n_rotate), dtype=int)
    cost = np.zeros((len(coords), n_rotate))
    inds = np.arange(len(coords), dtype=int)

    # generate rotations and resampling array!
    msg, warned = '', False
    for n in range(n_rotate):
        count, duplicated = 0, True

        if verbose:
            msg = 'Generating spin {:>5} of {:>5}'.format(n, n_rotate)
            print(msg, end='\r', flush=True)

        while duplicated and count < 500:
            count, duplicated = count + 1, False
            resampled = np.zeros(len(coords), dtype='int32')

            # rotate each hemisphere separately
            for h, rot in enumerate(_gen_rotation(seed=seed)):
                hinds = (hemiid == h)
                coor = coords[hinds]
                if len(coor) == 0:
                    continue

                # if we need an "exact" mapping (i.e., each node needs to be
                # assigned EXACTLY once) then we have to calculate the full
                # distance matrix which is a nightmare with respect to memory
                # for anything that isn't parcellated data.
                # that is, don't do this with vertex coordinates!
                if method == 'vasa':
                    dist = spatial.distance_matrix(coor, coor @ rot)
                    # min of max a la Vasa et al., 2018
                    col = np.zeros(len(coor), dtype='int32')
                    for _ in range(len(dist)):
                        # find parcel whose closest neighbor is farthest away
                        # overall; assign to that
                        row = dist.min(axis=1).argmax()
                        col[row] = dist[row].argmin()
                        cost[inds[hinds][row], n] = dist[row, col[row]]
                        # set to -inf and inf so they can't be assigned again
                        dist[row] = -np.inf
                        dist[:, col[row]] = np.inf
                # optimization of total cost using Hungarian algorithm. this
                # may result in certain parcels having higher cost than with
                # `method='vasa'` but should always result in the total cost
                # being lower #tradeoffs
                elif method == 'hungarian':
                    dist = spatial.distance_matrix(coor, coor @ rot)
                    row, col = optimize.linear_sum_assignment(dist)
                    cost[hinds, n] = dist[row, col]
                # if nodes can be assigned multiple targets, we can simply use
                # the absolute minimum of the distances (no optimization
                # required) which is _much_ lighter on memory
                # huge thanks to https://stackoverflow.com/a/47779290 for this
                # memory-efficient method
                elif method == 'original':
                    dist, col = spatial.cKDTree(coor @ rot).query(coor, 1)
                    cost[hinds, n] = dist

                resampled[hinds] = inds[hinds][col]

            # if we want to check for duplicates ensure that we don't have any
            if check_duplicates:
                if np.any(np.all(resampled[:, None] == spinsamples[:, :n], 0)):
                    duplicated = True
                # if our "spin" is identical to the input then that's no good
                elif np.all(resampled == inds):
                    duplicated = True

        # if we broke out because we tried 500 rotations and couldn't generate
        # a new one, warn that we're using duplicate rotations and give up.
        # this should only be triggered if check_duplicates is set to True
        if count == 500 and not warned:
            warnings.warn(
                'Duplicate rotations used. Check resampling array '
                'to determine real number of unique permutations.', stacklevel=2)
            warned = True

        spinsamples[:, n] = resampled

    if verbose:
        print(' ' * len(msg) + '\b' * len(msg), end='', flush=True)

    if return_cost:
        return spinsamples, cost

    return spinsamples


def get_dominance_stats(X, y, use_adjusted_r_sq=True, verbose=False, n_jobs=1):
    """
    Return the dominance analysis statistics for multilinear regression.

    This is a rewritten & simplified version of [DA1]_. It is briefly
    tested against the original package, but still in early stages.
    Please feel free to report any bugs.

    Warning: Still work-in-progress. Parameters might change!

    Parameters
    ----------
    X : (N, M) array_like
        Input data
    y : (N,) array_like
        Target values
    use_adjusted_r_sq : bool, optional
        Whether to use adjusted r squares. Default: True
    verbose : bool, optional
        Whether to print debug messages. Default: False
    n_jobs : int, optional
        The number of jobs to run in parallel. Default: 1

    Returns
    -------
    model_metrics : dict
        The dominance metrics, currently containing `individual_dominance`,
        `partial_dominance`, `total_dominance`, and `full_r_sq`.
    model_r_sq : dict
        Contains all model r squares

    Notes
    -----
    Example usage

    .. code:: python

        from netneurotools.stats import get_dominance_stats
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)
        model_metrics, model_r_sq = get_dominance_stats(X, y)

    To compare with [DA1]_, use `use_adjusted_r_sq=False`

    .. code:: python

        from dominance_analysis import Dominance_Datasets
        from dominance_analysis import Dominance
        boston_dataset=Dominance_Datasets.get_boston()
        dominance_regression=Dominance(data=boston_dataset,
                                       target='House_Price',objective=1)
        incr_variable_rsquare=dominance_regression.incremental_rsquare()
        dominance_regression.dominance_stats()

    References
    ----------
    .. [DA1] https://github.com/dominance-analysis/dominance-analysis

    """
    # this helps to remove one element from a tuple
    def remove_ret(tpl, elem):
        lst = list(tpl)
        lst.remove(elem)
        return tuple(lst)

    # sklearn linear regression wrapper
    def get_reg_r_sq(X, y, use_adjusted_r_sq=True):
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        yhat = lin_reg.predict(X)
        SS_Residual = sum((y - yhat) ** 2)
        SS_Total = sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (float(SS_Residual)) / SS_Total
        adjusted_r_squared = 1 - (1 - r_squared) * \
            (len(y) - 1) / (len(y) - X.shape[1] - 1)
        if use_adjusted_r_sq:
            return adjusted_r_squared
        else:
            return r_squared

    # helper function to compute r_sq for a given idx_tuple
    def compute_r_sq(idx_tuple):
        return idx_tuple, get_reg_r_sq(X[:, idx_tuple],
                                       y,
                                       use_adjusted_r_sq=use_adjusted_r_sq)

    # generate all predictor combinations in list (num of predictors) of lists
    n_predictor = X.shape[-1]
    # n_comb_len_group = n_predictor - 1
    predictor_combs = [list(combinations(range(n_predictor), i))
                       for i in range(1, n_predictor + 1)]
    if verbose:
        print(f"[Dominance analysis] Generated \
              {len([v for i in predictor_combs for v in i])} combinations")

    model_r_sq = dict()
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_r_sq)(idx_tuple)
        for len_group in tqdm(predictor_combs,
                              desc='num-of-predictor loop',
                              disable=not verbose)
        for idx_tuple in tqdm(len_group,
                              desc='insider loop',
                              disable=not verbose))

    # extract r_sq from results
    for idx_tuple, r_sq in results:
        model_r_sq[idx_tuple] = r_sq

    if verbose:
        print(f"[Dominance analysis] Acquired {len(model_r_sq)} r^2's")

    # getting all model metrics
    model_metrics = dict([])

    # individual dominance
    individual_dominance = []
    for i_pred in range(n_predictor):
        individual_dominance.append(model_r_sq[(i_pred,)])
    individual_dominance = np.array(individual_dominance).reshape(1, -1)
    model_metrics["individual_dominance"] = individual_dominance

    # partial dominance
    partial_dominance = [[] for _ in range(n_predictor - 1)]
    for i_len in range(n_predictor - 1):
        i_len_combs = list(combinations(range(n_predictor), i_len + 2))
        for j_node in range(n_predictor):
            j_node_sel = [v for v in i_len_combs if j_node in v]
            reduced_list = [remove_ret(comb, j_node) for comb in j_node_sel]
            diff_values = [
                model_r_sq[j_node_sel[i]] - model_r_sq[reduced_list[i]]
                for i in range(len(reduced_list))]
            partial_dominance[i_len].append(np.mean(diff_values))

    # save partial dominance
    partial_dominance = np.array(partial_dominance)
    model_metrics["partial_dominance"] = partial_dominance
    # get total dominance
    total_dominance = np.mean(
        np.r_[individual_dominance, partial_dominance], axis=0)
    # test and save total dominance
    assert np.allclose(total_dominance.sum(),
                       model_r_sq[tuple(range(n_predictor))]), \
           "Sum of total dominance is not equal to full r square!"
    model_metrics["total_dominance"] = total_dominance
    # save full r^2
    model_metrics["full_r_sq"] = model_r_sq[tuple(range(n_predictor))]

    return model_metrics, model_r_sq


def network_pearsonr(annot1, annot2, weight):
    r"""
    Calculate pearson correlation between two annotation vectors.

    .. warning::
       Test before use.

    Parameters
    ----------
    annot1 : (N,) array_like
        First annotation vector, demean will be applied.
    annot2 : (N,) array_like
        Second annotation vector, demean will be applied.
    weight : (N, N) array_like
        Weight matrix. Diagonal elements should be 1.

    Returns
    -------
    corr : float
        Network correlation between `annot1` and `annot2`

    Notes
    -----
    If Pearson correlation is represented as

    .. math::
        \rho_{x,y} = \dfrac{
            \mathrm{sum}(I \times (\hat{x} \otimes \hat{y}))
        }{
            \sigma_x \sigma_y
        }

    The network correlation is defined analogously as

    .. math::
        \rho_{x,y,G} = \dfrac{
            \mathrm{sum}(W \times (\hat{x} \otimes \hat{y}))
        }{
            \sigma_{x,W} \sigma_{y,W}
        }

    where :math:`\hat{x}` and :math:`\hat{y}` are the demeaned annotation vectors,

    The weight matrix :math:`W` is used to represent the network structure.
    It is usually in the form of :math:`W = \\exp(-kL)` where :math:`L` is the
    length matrix and :math:`k` is a decay parameter.

    Example using shortest path length as weight

    .. code:: python

        spl, _ = distance_wei_floyd(D) # input should be distance matrix
        spl_wei = 1 / np.exp(spl)
        netcorr = network_pearsonr(annot1, annot2, spl_wei)

    Example using (inverse) effective resistance as weight

    .. code:: python

        R_eff = effective_resistance(W)
        R_eff_norm = R_eff / np.max(R_eff)
        W = 1 / R_eff_norm
        W = W / np.max(W)
        np.fill_diagonal(W, 1.0)
        netcorr = network_pearsonr(annot1, annot2, W)

    References
    ----------
    .. [1] Coscia, M. (2021). Pearson correlations on complex networks.
       Journal of Complex Networks, 9(6), cnab036.
       https://doi.org/10.1093/comnet/cnab036


    See Also
    --------
    netneurotools.stats.network_pearsonr_pairwise
    """
    annot1 = annot1 - np.mean(annot1)
    annot2 = annot2 - np.mean(annot2)
    upper = np.sum(np.multiply(weight, np.outer(annot1, annot2)))
    lower1 = np.sum(np.multiply(weight, np.outer(annot1, annot1)))
    lower2 = np.sum(np.multiply(weight, np.outer(annot2, annot2)))
    return upper / np.sqrt(lower1) / np.sqrt(lower2)


def network_pearsonr_numba(annot1, annot2, weight):
    """
    Numba version of :meth:`netneurotools.stats.network_pearsonr`.

    .. warning::
       Test before use.

    Parameters
    ----------
    annot1 : (N,) array_like
        First annotation vector, demean will be applied.
    annot2 : (N,) array_like
        Second annotation vector, demean will be applied.
    weight : (N, N) array_like
        Weight matrix. Diagonal elements should be 1.

    Returns
    -------
    corr : float
        Network correlation between `annot1` and `annot2`
    """
    n = annot1.shape[0]
    annot1 = annot1 - np.mean(annot1)
    annot2 = annot2 - np.mean(annot2)
    upper, lower1, lower2 = 0.0, 0.0, 0.0
    for i in range(n):
        for j in range(n):
            upper += annot1[i] * annot2[j] * weight[i, j]
            lower1 += annot1[i] * annot1[j] * weight[i, j]
            lower2 += annot2[i] * annot2[j] * weight[i, j]
    return upper / np.sqrt(lower1) / np.sqrt(lower2)


if use_numba:
    network_pearsonr_numba = njit(network_pearsonr_numba)


def _cross_outer(annot_mat):
    """
    Calculate cross outer product of input matrix.

    This functions is only used in `network_pearsonr_pairwise`.

    Parameters
    ----------
    annot_mat : (N, D) array_like
        Input matrix

    Returns
    -------
    cross_outer : (N, N, D, D) numpy.ndarray
        Cross outer product of `annot_mat`
    """
    n_samp, n_feat = annot_mat.shape
    cross_outer = np.empty((n_samp, n_samp, n_feat, n_feat), annot_mat.dtype)
    for a in range(n_samp):
        for b in range(n_samp):
            for c in range(n_feat):
                for d in range(n_feat):
                    cross_outer[a, b, c, d] = annot_mat[a, c] * annot_mat[b, d]
    return cross_outer


if use_numba:
    # ("float64[:,:,:,::1](float64[:,::1])")
    _cross_outer = njit(_cross_outer)


def _multiply_sum(cross_outer, weight):
    """
    Multiply and sum cross outer product.

    This functions is only used in `network_pearsonr_pairwise`.

    Parameters
    ----------
    cross_outer : (N, N, D, D) array_like
        Cross outer product of `annot_mat`
    weight : (D, D) array_like
        Weight matrix

    Returns
    -------
    cross_outer_after : (N, N) numpy.ndarray
        Result of multiplying and summing `cross_outer`
    """
    n_samp, _, n_dim, _ = cross_outer.shape
    cross_outer_after = np.empty((n_samp, n_samp), cross_outer.dtype)
    for i in range(n_samp):
        for j in range(n_samp):
            curr_sum = 0.0
            for k in range(n_dim):
                for l in range(n_dim):  # noqa: E741
                    curr_sum += weight[k, l] * cross_outer[i, j, k, l]
            cross_outer_after[i, j] = curr_sum
    return cross_outer_after


if use_numba:
    # ("float64[:,::1](float64[:,:,:,::1],float64[:,::1])")
    _multiply_sum = njit(_multiply_sum)


def network_pearsonr_pairwise(annot_mat, weight):
    """
    Calculate pairwise network correlation between rows of `annot_mat`.

    .. warning::
       Test before use.

    Parameters
    ----------
    annot_mat : (N, D) array_like
        Input matrix
    weight : (D, D) array_like
        Weight matrix. Diagonal elements should be 1.

    Returns
    -------
    corr_mat : (N, N) numpy.ndarray
        Pairwise network correlation matrix

    Notes
    -----
    This is a faster version of :meth:`netneurotools.stats.network_pearsonr`
    for calculating pairwise network correlation between rows of `annot_mat`.
    Check :meth:`netneurotools.stats.network_pearsonr` for details.

    See Also
    --------
    netneurotools.stats.network_pearsonr
    """
    annot_mat_demean = annot_mat - np.mean(annot_mat, axis=1, keepdims=True)
    if use_numba:
        cross_outer = _cross_outer(annot_mat_demean)
        cross_outer_after = _multiply_sum(cross_outer, weight)
    else:
        # https://stackoverflow.com/questions/24839481/python-matrix-outer-product
        cross_outer = np.einsum('ac,bd->abcd', annot_mat_demean, annot_mat_demean)
        cross_outer_after = np.sum(np.multiply(cross_outer, weight), axis=(2, 3))
    # translating the two lines below in numba does not speed up much
    lower = np.sqrt(np.diagonal(cross_outer_after))
    return cross_outer_after / np.einsum('i,j', lower, lower)


def _onehot_quadratic_form_broadcast(Q_star):
    """
    Calculate one-hot quadratic form of input matrix.

    This functions is only used in `effective_resistance`.

    Parameters
    ----------
    Q_star : (N, N) array_like
        Input matrix

    Returns
    -------
    R_eff : (N, N) numpy.ndarray
        One-hot quadratic form of `Q_star`
    """
    n = Q_star.shape[0]
    R_eff = np.empty((n, n), Q_star.dtype)
    for i in range(n):
        for j in range(n):
            R_eff[i, j] = Q_star[i, i] - Q_star[j, i] - Q_star[i, j] + Q_star[j, j]
    return R_eff


if use_numba:
    # ("float64[:,::1](float64[:,::1])")
    _onehot_quadratic_form_broadcast = njit(_onehot_quadratic_form_broadcast)


def effective_resistance(W, directed=True):
    """
    Calculate effective resistance matrix.

    The effective resistance between two nodes in a graph, often used in the context
    of electrical networks, is a measure that stems from the inverse of the Laplacian
    matrix of the graph.

    .. warning::
       Test before use.

    Parameters
    ----------
    W : (N, N) array_like
        Weight matrix.
    directed : bool, optional
        Whether the graph is directed. This is used to determine whether to turn on
        the :code:`hermitian=True` option in :func:`numpy.linalg.pinv`. When you are
        using a symmetric weight matrix (while real-valued implying hermitian), you
        can set this to False for better performance. Default: True

    Returns
    -------
    R_eff : (N, N) numpy.ndarray
        Effective resistance matrix

    Notes
    -----
    The effective resistance between two nodes :math:`i` and :math:`j` is defined as

    .. math::
        R_{ij} = (e_i - e_j)^T Q^* (e_i - e_j)

    where :math:`Q^*` is the Moore-Penrose pseudoinverse of the Laplacian matrix
    :math:`L` of the graph, and :math:`e_i` is the :math:`i`-th standard basis vector.

    References
    ----------
    .. [1] Ellens, W., Spieksma, F. M., Van Mieghem, P., Jamakovic, A., & Kooij,
       R. E. (2011). Effective graph resistance. Linear Algebra and Its Applications,
       435(10), 2491–2506. https://doi.org/10.1016/j.laa.2011.02.024

    See Also
    --------
    netneurotools.stats.network_polarisation
    """
    L = _graph_laplacian(W)
    Q_star = np.linalg.pinv(L, hermitian=not directed)
    if use_numba:
        R_eff = _onehot_quadratic_form_broadcast(Q_star)
    else:
        Q_star_diag = np.diag(Q_star)
        R_eff = \
            Q_star_diag[:, np.newaxis] \
            - Q_star \
            - Q_star.T \
            + Q_star_diag[np.newaxis, :]
    return R_eff


def _polariz_diff(vec):
    """
    Calculate difference between positive and negative parts of a vector.

    This functions is only used in `network_polarisation`.

    Parameters
    ----------
    vec : (N,) array_like
        Input vector. Must have both positive and negative values.

    Returns
    -------
    vec_diff : (N,) numpy.ndarray
        Difference between positive and negative parts of `vec`
    """
    #
    vec_pos = np.maximum(vec, 0.0)
    vec_pos /= np.max(vec_pos)
    #
    vec_neg = np.minimum(vec, 0.0)
    vec_neg = np.abs(vec_neg)
    vec_neg /= np.max(vec_neg)
    return (vec_pos - vec_neg)


if use_numba:
    _polariz_diff = njit(_polariz_diff)


def _quadratic_form(W, vec_left, vec_right, squared=False):
    """
    Calculate quadratic form :math:`v_{left}^T W v_{right}`.

    Parameters
    ----------
    W : (N, N) array_like
        Input matrix.
    vec_left : (N,) array_like
        Left weight vector.
    vec_right : (N,) array_like
        Right weight vector.
    squared : bool, optional
        Whether to square the input weight matrix. Default: False

    Returns
    -------
    quadratic_form : float
        Quadratic form from `W`, `vec_left`, and `vec_right`
    """
    # [numpy]

    # (vec_left.T @ W @ vec_right)[0, 0]
    # [numba]
    # vec = np.ascontiguousarray(vec[np.newaxis, :])
    n = W.shape[0]
    ret = 0.0
    for i in range(n):
        for j in range(n):
            if squared:
                ret += vec_left[i] * vec_right[j] * W[i, j]**2
            else:
                ret += vec_left[i] * vec_right[j] * W[i, j]
    return ret


if use_numba:
    _quadratic_form = njit(_quadratic_form)


def network_polarisation(vec, W, directed=True):
    r"""
    Calculate polarisation of a vector on a graph.

    Network polarisation is a measure of polizzartion taken into account all the
    three factors below [1]_:

    - how extreme the opinions of the people are
    - how much they organize into echo chambers, and
    - how these echo chambers organize in the network

    .. warning::
       Test before use.

    Parameters
    ----------
    vec : (N,) array_like
        Polarization vector. Must have both positive and negative values. Will be
        normalized between -1 and 1 internally.
    W : (N, N) array_like
        Weight matrix.
    directed : bool, optional
        Whether the graph is directed. This is used to determine whether to turn on
        the :code:`hermitian=True` option in :func:`numpy.linalg.pinv`. When you are
        using a symmetric weight matrix (while real-valued implying hermitian), you
        can set this to False for better performance. Default: True

    Returns
    -------
    polariz : float
        Polarization of `vec` on `W`

    Notes
    -----
    The measure is based on the genralized Eucledian distance, defined as

    .. math::
        \delta_{G, o} = \sqrt{(o^+ - o^-)^T Q^* (o^+ - o^-)}

    where :math:`o^+` and :math:`o^-` are the positive and negative parts of the
    polarization vector, and :math:`Q^*` is the Moore-Penrose pseudoinverse
    of the Laplacian matrix :math:`L` of the graph. Check :func:`effective_resistance`
    for similarity.

    References
    ----------
    .. [1] Hohmann, M., Devriendt, K., & Coscia, M. (2023). Quantifying ideological
       polarization on a network using generalized Euclidean distance. Science Advances,
       9(9), eabq2044. https://doi.org/10.1126/sciadv.abq2044

    See Also
    --------
    netneurotools.stats.effective_resistance
    """
    L = _graph_laplacian(W)
    Q_star = np.linalg.pinv(L, hermitian=not directed)
    diff = _polariz_diff(vec)
    if use_numba:
        polariz_sq = _quadratic_form(Q_star, diff, diff, squared=False)
    else:
        polariz_sq = (diff.T @ Q_star @ diff)
    return np.sqrt(polariz_sq)


def network_variance(vec, D):
    r"""
    Calculate variance of a vector on a graph.

    Network variance is a measure of variance taken into account the network
    structure.

    .. warning::
       Test before use.

    Parameters
    ----------
    vec : (N,) array_like
        Input vector. Must be all positive.
        Will be normalized internally as a probability distribution.
    D : (N, N) array_like
        Distance matrix.

    Returns
    -------
    network_variance : float
        Network variance of `vec` on `D`

    Notes
    -----
    The network variance is defined as

    .. math::
        var(p) = \frac{1}{2} \sum_{i, j} p(i) p(j) d^2(i,j)

    where :math:`p` is the probability distribution of `vec`, and :math:`d(i,j)`
    is the distance between node :math:`i` and :math:`j`.

    The distance matrix :math:`D` can make use of effective resistance or its
    square root.

    Example using effective resistance as weight matrix

    .. code:: python

        R_eff = effective_resistance(W)
        netvar = network_variance(vec, R_eff)

    References
    ----------
    .. [1] Devriendt, K., Martin-Gutierrez, S., & Lambiotte, R. (2022).
       Variance and covariance of distributions on graphs. SIAM Review, 64(2),
       343–359. https://doi.org/10.1137/20M1361328

    See Also
    --------
    netneurotools.stats.network_covariance
    """
    p = vec / np.sum(vec)
    return 0.5 * (p.T @ np.multiply(D, D) @ p)


def network_variance_numba(vec, D):
    """
    Numba version of :meth:`netneurotools.stats.network_variance`.

    Network variance is a measure of variance taken into account the network
    structure.

    .. warning::
       Test before use.

    Parameters
    ----------
    vec : (N,) array_like
        Input vector. Must be all positive.
        Will be normalized internally as a probability distribution.
    D : (N, N) array_like
        Distance matrix.

    Returns
    -------
    network_variance : float
        Network variance of `vec` on `D`
    """
    p = vec / np.sum(vec)
    return 0.5 * _quadratic_form(D, p, p, squared=True)


if use_numba:
    network_variance_numba = njit(network_variance_numba)


def network_covariance(joint_pmat, D, calc_marginal=True):
    r"""
    Calculate covariance of a joint probability matrix on a graph.

    .. warning::
       Test before use.

    Parameters
    ----------
    joint_pmat : (N, N) array_like
        Joint probability matrix. Please make sure that it is valid.
    D : (N, N) array_like
        Distance matrix.
    calc_marginal : bool, optional
        Whether to calculate marginal variance. It will be marginally faster if
        :code:`calc_marginal=False` (returning marginal variances as 0). Default: True

    Returns
    -------
    network_covariance : float
        Covariance of `joint_pmat` on `D`
    var_p : float
        Marginal variance of `joint_pmat` on `D`.
        Will be 0 if :code:`calc_marginal=False`
    var_q : float
        Marginal variance of `joint_pmat` on `D`.
        Will be 0 if :code:`calc_marginal=False`

    Notes
    -----
    The network variance is defined as

    .. math::
        cov(P) = \frac{1}{2} \sum_{i, j} [p(i) q(j) - P(i,j)] d^2(i,j)

    where :math:`P` is the joint probability matrix, :math:`p` and :math:`q`
    are the marginal probability distributions of `joint_pmat`, and :math:`d(i,j)`
    is the distance between node :math:`i` and :math:`j`.

    Check :func:`network_variance` for usage.

    References
    ----------
    .. [1] Devriendt, K., Martin-Gutierrez, S., & Lambiotte, R. (2022).
       Variance and covariance of distributions on graphs. SIAM Review, 64(2),
       343–359. https://doi.org/10.1137/20M1361328

    See Also
    --------
    netneurotools.stats.network_variance
    """
    p = np.sum(joint_pmat, axis=1)
    q = np.sum(joint_pmat, axis=0)
    D_sq = np.multiply(D, D)
    cov = p.T @ D_sq @ q - np.sum(np.multiply(joint_pmat, D_sq))
    if calc_marginal:
        var_p = p.T @ D_sq @ p
        var_q = q.T @ D_sq @ q
    else:
        var_p, var_q = 0, 0
    return 0.5 * cov, 0.5 * var_p, 0.5 * var_q


def network_covariance_numba(joint_pmat, D, calc_marginal=True):
    """
    Numba version of :meth:`netneurotools.stats.network_covariance`.

    .. warning::
       Test before use.

    Parameters
    ----------
    joint_pmat : (N, N) array_like
        Joint probability matrix. Please make sure that it is valid.
    D : (N, N) array_like
        Distance matrix.
    calc_marginal : bool, optional
        Whether to calculate marginal variance. It will be marginally faster if
        :code:`calc_marginal=False` (returning marginal variances as 0). Default: True

    Returns
    -------
    network_covariance : float
        Covariance of `joint_pmat` on `D`
    var_p : float
        Marginal variance of `joint_pmat` on `D`.
        Will be 0 if :code:`calc_marginal=False`
    var_q : float
        Marginal variance of `joint_pmat` on `D`.
        Will be 0 if :code:`calc_marginal=False`
    """
    n = joint_pmat.shape[0]
    p = np.sum(joint_pmat, axis=1)
    q = np.sum(joint_pmat, axis=0)
    cov = 0.0
    var_p, var_q = 0.0, 0.0
    for i in range(n):
        for j in range(n):
            cov += (p[i] * q[j] - joint_pmat[i, j]) * D[i, j]**2
            if calc_marginal:
                var_p += p[i] * p[j] * D[i, j]**2
                var_q += q[i] * q[j] * D[i, j]**2
    return 0.5 * cov, 0.5 * var_p, 0.5 * var_q


if use_numba:
    network_covariance_numba = njit(network_covariance_numba)
