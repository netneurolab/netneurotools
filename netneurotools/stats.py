# -*- coding: utf-8 -*-
"""
Functions for performing statistical preprocessing and analyses
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zmap
from sklearn.utils.validation import check_random_state

from . import utils


def residualize(X, Y, Xc=None, Yc=None, normalize=True, add_intercept=True):
    """
    Returns residuals of `Y ~ X`

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

    Notes
    -----
    If both `Xc` and `Yc` are provided, these are used to calculate betas which
    are then applied to `X` and `Y`.
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
    >>> from netneurotools.stats import get_mad_outliers
    >>> X = np.array([[0, 5, 10, 15], [1, 4, 11, 16], [20, 20, 20, 20]])
    >>> outliers = get_mad_outliers(X)
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


def perm_1samp(data, n_perm=1000):
    """
    Non-parametric equivalent of :py:func:`scipy.stats.ttest_1samp`

    Generates null distribution of `data` via sign flipping and regeneration of
    mean

    Parameters
    ----------
    data : (N, M) array_like
        Where `N` is samples and `M` is features

    Returns
    -------
    permutations : (M, P)
        Null distribution for each of `M` features from `data`
    """

    permutations = np.zeros((data.shape[-1], n_perm))

    for perm in range(n_perm):
        flip = np.random.choice([-1, 1], size=data.shape)
        permutations[:, perm] = np.mean(data * flip, axis=0)

    return permutations


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

    >>> from netneurotools.tests import make_correlated_xy
    >>> x, y = make_correlated_xy(size=(68,))
    >>> x.shape, y.shape
    ((68,), (68,))

    We can correlate the vectors to see how related they are:

    >>> from scipy.stats import pearsonr
    >>> r, p = pearsonr(x, y)
    >>> r, p
    (0.6961556744296582, 4.376432884386363e-11)

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

    >>> from netneurotools.utils import get_cammoun2012_info
    >>> coords, hemi = get_cammoun2012_info(scale=33)
    >>> coords.shape, hemi.shape
    ((68, 3), (68,))

    Next, we generate a resampling array based on this "rotation" concept:

    >>> from netneurotools.stats import gen_spinsamples
    >>> spin = gen_spinsamples(coords, hemi)
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
