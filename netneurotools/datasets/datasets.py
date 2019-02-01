# -*- coding: utf-8 -*-
"""
Dataset fetcher / creation / whathaveyou
"""

import pickle
from pkg_resources import resource_filename
import numpy as np
from sklearn.utils.validation import check_random_state


def make_correlated_xy(corr=0.85, size=10000, seed=None, tol=0.001):
    """
    Generates random vectors that are correlated to approximately `corr`

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
        `corr`. Default: 0.05

    Returns
    -------
    vectors : numpy.ndarray
        Random vectors of size `size` with correlation specified by `corr`

    Examples
    --------
    >>> from netneurotools import datasets

    By default two vectors are generated with specified correlation

    >>> x, y = datasets.make_correlated_xy()
    >>> np.corrcoef(x, y)
    array([[1.        , 0.85083661],
           [0.85083661, 1.        ]])
    >>> x, y = datasets.make_correlated_xy(corr=0.2)
    >>> np.corrcoef(x, y)
    array([[1.        , 0.20069953],
           [0.20069953, 1.        ]])

    You can also provide correlation matrices to generate more than two vectors
    if desired. Note that this makes it more difficult to ensure the actual
    correlations are close to the desired values:

    >>> corr = [[1, 0.5, 0.3], [0.5, 1, 0], [0.3, 0, 1]]
    >>> out = datasets.make_correlated_xy(corr=corr)
    >>> out.shape
    (3, 10000)
    >>> np.corrcoef(out)
    array([[1.        , 0.50965273, 0.30235686],
           [0.50965273, 1.        , 0.01089107],
           [0.30235686, 0.01089107, 1.        ]])
    """

    rs = check_random_state(seed)

    # no correlations outside [-1, 1] bounds
    if np.any(np.abs(corr) > 1):
        raise ValueError('Provided `corr` must (all) be in range [-1, 1].')

    # if we're given a single number, assume two vectors are desired
    if isinstance(corr, (int, float)):
        covs = np.ones((2, 2)) * 0.111
        covs[(0, 1), (1, 0)] *= corr
    # if we're given a correlation matrix, assume `N` vectors are desired
    elif isinstance(corr, (list, np.ndarray)):
        corr = np.asarray(corr)
        if corr.ndim != 2 or len(corr) != len(corr.T):
            raise ValueError('If `corr` is a list or array, must be a 2D '
                             'square array, not {}'.format(corr.shape))
        if np.any(np.diag(corr) != 1):
            raise ValueError('Diagonal of `corr` must be 1.')
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


def load_cammoun2012(scale, surface=True):
    """
    Returns centroids / hemi assignment of parcels from Cammoun et al., 2012

    Centroids are defined on the spherical projection of the fsaverage cortical
    surface reconstruciton (FreeSurfer v6.0.1)

    Parameters
    ----------
    scale : {33, 60, 125, 250, 500}
        Scale of parcellation for which to get centroids / hemisphere
        assignments
    surface : bool, optional
        Whether to return coordinates from surface instead of volume
        reconstruction. Default: True

    Returns
    -------
    centroids : (N, 3) numpy.ndarray
        Centroids of parcels defined by Cammoun et al., 2012 parcellation
    hemiid : (N,) numpy.ndarray
        Hemisphere assignment of `centroids`, where 0 indicates left and 1
        indicates right hemisphere

    References
    ----------
    Cammoun, L., Gigandet, X., Meskaldji, D., Thiran, J. P., Sporns, O., Do, K.
    Q., Maeder, P., and Meuli, R., & Hagmann, P. (2012). Mapping the human
    connectome at multiple scales with diffusion spectrum MRI. Journal of
    Neuroscience Methods, 203(2), 386-397.

    Examples
    --------
    >>> from netneurotools import datasets

    >>> coords, hemiid = datasets.load_cammoun2012(scale=33)
    >>> coords.shape, hemiid.shape
    ((68, 3), (68,))

    ``hemiid`` is a vector of 0 and 1 denoting which ``coords`` are in the
    left / right hemisphere, respectively:

    >>> hemiid
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1])
    """

    pckl = resource_filename('netneurotools', 'data/cammoun.pckl')

    if not isinstance(scale, int):
        try:
            scale = int(scale)
        except ValueError:
            raise ValueError('Provided `scale` must be integer in [33, 60, '
                             '125, 250, 500], not {}'.format(scale))
    if scale not in [33, 60, 125, 250, 500]:
        raise ValueError('Provided `scale` must be integer in [33, 60, 125, '
                         '250, 500], not {}'.format(scale))

    with open(pckl, 'rb') as src:
        data = pickle.load(src)['cammoun{}'.format(str(scale))]

    return data['centroids'], data['hemiid']
