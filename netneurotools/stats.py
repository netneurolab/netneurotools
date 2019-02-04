# -*- coding: utf-8 -*-
"""
Functions for performing statistical preprocessing and analyses
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zmap
from scipy.stats.stats import _chk_asarray, _chk2_asarray
from sklearn.utils.validation import check_random_state

from . import utils


def residualize(X, Y, Xc=None, Yc=None, normalize=True, add_intercept=True):
    """
    Returns residuals of regression equation from `Y ~ X`

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
    Yr = Y - (X @ betas)
    Ycr = Yc - (Xc @ betas)

    if normalize:
        Yr = zmap(Yr, compare=Ycr)

    return Yr


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
    Non-parametric equivalent of :py:func:`scipy.stats.ttest_1samp`

    Generates two-tailed p-value for hypothesis of whether `a` differs from
    `popmean`

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
    pvalue : float or numpy.ndarray
        Non-parametric p-value

    Notes
    -----
    Providing multiple values to `popmean` to run *independent* tests in
    parallel is not currently supported.

    Examples
    --------
    >>> from netneurotools import stats
    >>> np.random.seed(7654567)  # set random seed for reproducible results
    >>> rvs = np.random.normal(loc=5, scale=10, size=(50, 2))

    Test if mean of random sample is equal to true mean, and different mean. We
    reject the null hypothesis in the second case and don't reject it in the
    first case.

    >>> stats.permtest_1samp(rvs, 5.0)
    array([0.48551449, 0.95904096])
    >>> stats.permtest_1samp(rvs, 0.0)
    array([0.00699301, 0.000999  ])

    Example using axis and non-scalar dimension for population mean

    >>> stats.permtest_1samp(rvs, [5.0, 0.0])
    array([0.48551449, 0.000999  ])
    >>> stats.permtest_1samp(rvs.T, [5.0, 0.0], axis=1)
    array([0.51548452, 0.000999  ])
    """

    a, axis = _chk_asarray(a, axis)
    rs = check_random_state(seed)

    # ensure popmean will broadcast to `a` correctly
    popmean = np.asarray(popmean)
    if popmean.ndim != a.ndim:
        popmean = np.expand_dims(popmean, axis=axis)

    # center `a` around `popmean` and calculate original mean
    zeroed = a - popmean
    true_mean = zeroed.mean(axis=axis)

    # array to hold counts; use 1s instead of 0s to account for original value
    permutations = np.ones(np.delete(a.shape, axis)) if axis is not None else 1

    # this for loop is not _the fastest_ but is memory efficient
    # the broadcasting alt. would mean storing zeroed.size * n_perm in memory
    for perm in range(n_perm):
        flipped = zeroed * rs.choice([-1, 1], size=zeroed.shape)  # sign flip
        permutations += np.abs(flipped.mean(axis=axis)) >= np.abs(true_mean)

    return permutations / (n_perm + 1)  # + 1 in denom accounts for true_mean


def permtest_rel(a, b, axis=0, n_perm=1000, seed=0):
    """
    Non-parametric equivalent of :py:func:`scipy.stats.ttest_rel`

    Generates two-tailed p-value for hypothesis of whether related samples `a`
    and `b` differ

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
    pvalue : float or numpy.ndarray
        Non-parametric p-value

    Examples
    --------
    >>> from netneurotools import stats

    >>> np.random.seed(12345678)  # set random seed for reproducible results
    >>> rvs1 = np.random.normal(loc=5, scale=10, size=500)
    >>> rvs2 = (np.random.normal(loc=5, scale=10, size=500)
    ...         + np.random.normal(scale=0.2, size=500))
    >>> stats.permtest_rel(rvs1, rvs2)
    0.8021978021978022

    >>> rvs3 = (np.random.normal(loc=8, scale=10, size=500)
    ...         + np.random.normal(scale=0.2, size=500))
    >>> stats.permtest_rel(rvs1, rvs3)
    0.000999000999000999
    """

    a, b, axis = _chk2_asarray(a, b, axis)
    rs = check_random_state(seed)

    # calculate original difference in means
    ab = np.stack([a, b], axis=0)
    true_diff = np.diff(ab, axis=0).squeeze().mean(axis=axis)

    # array to hold counts; use 1s instead of 0s to account for original value
    permutations = np.ones(np.delete(a.shape, axis)) if axis is not None else 1

    # idx array
    reidx = np.meshgrid(*[range(f) for f in ab.shape], indexing='ij')

    for perm in range(n_perm):
        # swap matched samples between `a` and `b` randomly and recompute diff
        reidx[0] = rs.random_sample(ab.shape).argsort(axis=0)
        perm_diff = np.diff(ab[tuple(reidx)], axis=0).squeeze().mean(axis=axis)
        permutations += np.abs(perm_diff) >= np.abs(true_diff)

    return permutations / (n_perm + 1)  # + 1 in denom accounts for true_diff


def _yield_rotations(n_rotate, seed=None):
    """
    Generates random matrix for rotating spherical coordinates

    Parameters
    ----------
    n_rotate : int
        Number of rotations to generate
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation

    Yields
    -------
    rotate_l, rotate_r : (3, 3) numpy.ndarray
        Rotations for left and right hemispheres, respectively
    """

    rs = check_random_state(seed)

    # for reflecting across Y-Z plane
    reflect = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    for n in range(n_rotate):
        # generate rotation for left
        rotate_l, temp = np.linalg.qr(rs.normal(size=(3, 3)))
        rotate_l = rotate_l @ np.diag(np.sign(np.diag(temp)))
        if np.linalg.det(rotate_l) < 0:
            rotate_l[:, 0] = -rotate_l[:, 0]

        # reflect the left rotation across Y-Z plane
        rotate_r = reflect @ rotate_l @ reflect

        yield rotate_l, rotate_r


def gen_spinsamples(coords, hemiid, n_rotate=1000, seed=None):
    """
    Generates resampling array for `coords` via rotational spins

    Using the method initially proposed in [ST1]_ (and later modified / updated
    based on findings in [ST2]_ and [ST3]_), this function can be used to
    generate a resampling array for conducting spatial permutation tests.

    Parameters
    ----------
    coords : (N, 3) array_like
        X, Y, Z coordinates of `N` nodes/parcels/regions/vertices defined on a
        sphere
    hemiid : (N,) array_like
        Array denoting hemisphere designation of coordinates in `coords`, where
        `hemiid=0` denotes the left and `hemiid=1` the right hemisphere
    n_rotate : int, optional
        Number of random rotations to generate. Default: 1000
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    permsamples : (N, `n_rotate`) numpy.ndarray
        Resampling matrix to use in permuting data based on supplied `coords`

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

    Examples
    --------
    Let's say we have two vectors of data, defined for a set of 68 brain
    regions:

    >>> from netneurotools import datasets
    >>> np.random.seed(1234)  # set random seed for reproducible results
    >>> x, y = datasets.make_correlated_xy(size=(68,))
    >>> x.shape, y.shape
    ((68,), (68,))

    We can correlate the vectors to see how related they are:

    >>> from scipy.stats import pearsonr
    >>> r, p = pearsonr(x, y)
    >>> r, p
    (0.8504392136292929, 4.441430678595365e-20)

    These vectors are quite correlated, and the correlation appears to be very
    significant. Unfortunately, there's a possibility that the correlation of
    these two vectors is inflated by the spatial organization of the brain. We
    want to create a null distribution of correlations via permutation to
    assess whether this correlation is truly significant or not.

    We could just randomly permute one of the vectors and regenerate the
    correlation:

    >>> r_perm = np.zeros((1000,))
    >>> for perm in range(1000):
    ...     r_perm[perm] = pearsonr(np.random.permutation(x), y)[0]
    >>> p_perm = (np.sum(r_perm > r) + 1) / (len(r_perm) + 1)
    >>> p_perm
    0.000999000999000999

    The permuted p-value suggests that our data are, indeed, highly correlated.
    Unfortunately this does not take into account that the data are constrained
    by a spatial toplogy (i.e., the brain) and thus are not entirely
    exchangeable as is assumed by a normal permutation test.

    Instead, we can resample the data by thinking about the brain as a sphere
    and considering random rotations of this sphere. If we rotate the data and
    resample datapoints based on their rotated values, we can generate a null
    distribution that is more appropriate.

    To do this we need the spatial coordinates of our brain regions as well as
    an array indicating to which hemisphere each region belongs. We'll use one
    of the parcellations commonly employed in the lab (Cammoun et al., 2012):

    >>> cammoun = datasets.load_cammoun2012(scale=33)
    >>> cammoun.coords.shape, cammoun.hemi.shape
    ((68, 3), (68,))

    Next, we generate a resampling array based on this "rotation" concept:

    >>> from netneurotools import stats
    >>> spin = stats.gen_spinsamples(cammoun.coords, cammoun.hemi)
    >>> spin.shape
    (68, 1000)

    We can use this to resample one of our data vectors and regenerate the
    correlations:

    >>> r_spinperm = np.zeros((1000,))
    >>> for perm in range(1000):
    ...     r_spinperm[perm] = pearsonr(x[spin[:, perm]], y)[0]
    >>> p_spinperm = (np.sum(r_perm > r) + 1) / (len(r_perm) + 1)
    >>> p_spinperm
    0.000999000999000999

    We see that the original correlation is still significant!
    """

    # check supplied coordinate shape
    if coords.shape[-1] != 3 or coords.squeeze().ndim != 2:
        raise ValueError('Provided `coords` must be of shape (N, 3), not {}'
                         .format(coords.shape))

    # ensure hemisphere designation array is correct
    hemiid = hemiid.squeeze()
    if hemiid.ndim != 1:
        raise ValueError('Provided `hemiid` array must be one-dimensional.')
    if len(coords) != len(hemiid):
        raise ValueError('Provided `coords` and `hemiid` must have the same '
                         'length. Provided lengths: coords = {}, hemiid = {}'
                         .format(len(coords), len(hemiid)))
    if not np.allclose(np.unique(hemiid), [0, 1]):
        raise ValueError('Hemiid must have values of [0, 1] denoting left '
                         'and right hemisphere coordinates, respectively. '
                         'Provided array contains values: {}'
                         .format(np.unique(hemiid)))

    # empty array to store resampling indices
    permsamples = np.zeros((len(coords), n_rotate), dtype=int)

    # split coordinates into left / right hemispheres
    inds_l, inds_r = np.where(hemiid == 0)[0], np.where(hemiid == 1)[0]
    coords_l, coords_r = coords[inds_l], coords[inds_r]

    # rotate coordinates and reassign indices!
    for n, (left, right) in enumerate(_yield_rotations(n_rotate, seed=seed)):
        # calculate Euclidean distance between original and rotated coords
        dist_l = cdist(coords_l, coords_l @ left)
        dist_r = cdist(coords_r, coords_r @ right)

        # find index of sample with minimum distance to original and assign
        permsamples[inds_l, n] = inds_l[dist_l.argmin(axis=1)]
        permsamples[inds_r, n] = inds_r[dist_r.argmin(axis=1)]

    return permsamples
