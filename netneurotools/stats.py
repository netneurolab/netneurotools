# -*- coding: utf-8 -*-
"""
Functions for performing statistical preprocessing and analyses
"""

import warnings

import numpy as np
from scipy import optimize
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


def _gen_rotation(seed=None):
    """
    Generates random matrix for rotating spherical coordinates

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


def gen_spinsamples(coords, hemiid, exact=False, n_rotate=1000, seed=None):
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
    exact : bool, optional
        Whether each node/parcel/region must be uniquely re-assigned in every
        rotation. Setting to True will drastically increase the runtime of this
        function! Default: False
    n_rotate : int, optional
        Number of random rotations to generate. Default: 1000
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    spinsamples : (N, `n_rotate`) numpy.ndarray
        Resampling matrix to use in permuting data based on supplied `coords`
    cost : (`n_rotate`,) numpy.ndarray
        Cost (in distance between node assignments) of each rotation in
        `spinsamples`

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
    """

    seed = check_random_state(seed)

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
    spinsamples = np.zeros((len(coords), n_rotate), dtype=int)
    cost = np.zeros(n_rotate)

    # split coordinates into left / right hemispheres
    inds_l, inds_r = np.where(hemiid == 0)[0], np.where(hemiid == 1)[0]
    coords_l, coords_r = coords[inds_l], coords[inds_r]

    # generate rotations and resampling array!
    warned = False
    for n in range(n_rotate):
        # try and avoid duplicates, if at all possible...
        count, duplicated = 0, True
        while duplicated and count < 500:
            count, duplicated = count + 1, False
            resampled = np.zeros(len(coords), dtype=int)

            # generate left + right hemisphere rotations
            left, right = _gen_rotation(seed=seed)

            # calculate Euclidean distance between original and rotated coords
            dist_l = cdist(coords_l, coords_l @ left)
            dist_r = cdist(coords_r, coords_r @ right)

            # find mapping of rotated coords to original coords
            # if we need an exact mapping (i.e., every node needs a unique
            # assignment in the rotation) then we need to use an optimization
            # procedure (i.e., the Hungarian algorithm)
            if exact:
                lrow, lcol = optimize.linear_sum_assignment(dist_l)
                rrow, rcol = optimize.linear_sum_assignment(dist_r)
            # otherwise, if nodes can be assigned multiple targets, we can just
            # use the absolute minimum of the distances.
            else:
                lrow, lcol = range(len(dist_l)), dist_l.argmin(axis=1)
                rrow, rcol = range(len(dist_r)), dist_r.argmin(axis=1)

            # generate resampling vector
            resampled[inds_l] = inds_l[lcol]
            resampled[inds_r] = inds_r[rcol]

            # check whether this is a duplicate resampling; if so, try again
            dupe_seq = resampled[:, None] == spinsamples[:, :n]
            if dupe_seq.all(axis=0).any():
                duplicated = True

        # if we broke out because we tried 500 rotations and couldn't generate
        # a new one, just warn that we're using duplicate rotations and give up
        if count == 500 and not warned:
            warnings.warn('WANRING: Duplicate rotations used. Check '
                          'resampling array to determine real number of '
                          'unique permutations.')
            warned = True

        # caclulate cost of rotation in terms of distance b/w node assignments
        cost[n] = dist_l[lrow, lcol].sum() + dist_r[rrow, rcol].sum()

        # store resampling vector
        spinsamples[:, n] = resampled

    return spinsamples, cost
