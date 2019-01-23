# -*- coding: utf-8 -*-
"""
Functions for performing statistical preprocessing and analyses
"""

import numpy as np
from scipy.spatial.distance import cdist
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

    if seed is not None:
        rs = np.random.RandomState(seed)
    else:
        rs = np.random

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
    Generates permutation resampling array via rotational spins

    Parameters
    ----------
    coords : (N, 3) array_like
        X, Y, Z coordinates of nodes on spherical surface
    hemiid : (N,) array_like
        Array denoting hemisphere designation of coordinates in `coords`, where
        `hemiid=0` is the left and `hemiid=1` is the right hemisphere
    n_rotate : int, optional
        Number of rotations to generate. Default: 1000
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation

    Returns
    -------
    permsamples : (N, `n_rotate`) numpy.ndarray
        Resampling matrix to use in permuting data based on supplied `coords`

    References
    ----------
    Alexander-Bloch, A., Shou, H., Liu, S., Satterthwaite, T. D., Glahn, D. C.,
    Shinohara, R. T., Vandekar, S. N., & Raznahan, A. (2018). On testing for
    spatial correspondence between maps of human brain structure and function.
    NeuroImage, 178, 540-51.
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

