"""Functions for calculating spatial statistics."""

import numpy as np

try:
    from numba import njit
    has_numba = True
except ImportError:
    has_numba = False


def _morans_i_vectorized(annot, weight):
    n = annot.shape[0]
    annot_demean = annot - np.mean(annot)
    W = np.sum(weight)
    upper = np.sum(weight * np.outer(annot_demean, annot_demean))
    lower = np.sum(annot_demean ** 2)
    return n * upper / (lower * W)


@njit
def _morans_i_numba(annot, weight):
    n = annot.shape[0]
    annot = annot - np.mean(annot)  # all occurances are demean-ed
    upper, lower, W = 0.0, 0.0, 0.0
    for i in range(n):
        lower += annot[i] ** 2
        for j in range(n):
            upper += weight[i, j] * annot[i] * annot[j]
            W += weight[i, j]
    return n * upper / lower / W


def morans_i(annot, weight, use_numba=has_numba):
    r"""
    Calculate Moran's I for spatial autocorrelation.

    Parameters
    ----------
    annot : array-like, shape (n,)
        Array of annotations to calculate Moran's I for.
    weight : array-like, shape (n, n)
        Spatial weight matrix. Note that we do not explicitly check for
        symmetry in the weight matrix, nor zero-diagonal elements.
    use_numba : bool, optional
        Whether to use numba for calculation. Default: True (if numba is
        installed).

    Returns
    -------
    morans_i : float
        Moran's I value for the given annotations and weight matrix.

    Notes
    -----
    Moran's I is calculated as:

    .. math::
        I = \frac{n}{\sum_{i=1}^{n} \sum_{j=1}^{n} w_{ij}} \frac{\sum_{i=1}^{n}
        \sum_{j=1}^{n} w_{ij} (x_i - \bar{x})(x_j - \bar{x})}{\sum_{i=1}^{n}
        (x_i - \bar{x})^2}

    where :math:`n` is the number of observations, :math:`w_{ij}` is the spatial
    weight between observations :math:`i` and :math:`j`, :math:`x_i` is the
    annotation for observation :math:`i`, and :math:`\bar{x}` is the mean
    annotation value.

    The value can be tested using the R pacakge ``spdep``:

    .. code:: R

        x <- rnorm(100)
        m <- matrix(runif(100*100), nrow=100)
        moran.test(x, mat2listw(m))

    See Also
    --------
    netneurotools.spatial.spatial_stats.local_morans_i
    """
    if use_numba:
        return _morans_i_numba(annot, weight)
    else:
        return _morans_i_vectorized(annot, weight)


def local_morans_i(annot, weight, use_sampvar=False):
    r"""
    Calculate local Moran's I for spatial autocorrelation.

    Parameters
    ----------
    annot : array-like, shape (n,)
        Array of annotations to calculate Moran's I for.
    weight : array-like, shape (n, n)
        Spatial weight matrix. Note that we do not explicitly check for symmetry
        in the weight matrix, nor zero-diagonal elements.
    use_sampvar : bool, optional
        Whether to use sample variance (n - 1) in calculation. Default: False.

    Returns
    -------
    local_morans_i : array, shape (n,)
        Local Moran's I values for the given annotations and weight matrix.

    Notes
    -----
    Local Moran's I is calculated as:

    .. math::
        I_i = \frac{(x_i - \bar{x})}{\sum_{k=1}^n (x_k - \bar{x})^2/(n-1)}
        \sum_{j=1}^n w_{ij}(x_j - \bar{x})

    where :math:`n` is the number of observations, :math:`w_{ij}` is the spatial
    weight between observations :math:`i` and :math:`j`, :math:`x_i` is the
    annotation for observation :math:`i`, and :math:`\bar{x}` is the mean
    annotation value.

    The value can be tested using the R pacakge ``spdep``:

    .. code:: R

            x <- rnorm(100)
            m <- matrix(runif(100*100), nrow=100)
            localmoran(v, mat2listw(m), mlvar=TRUE)

    See Also
    --------
    netneurotools.spatial.spatial_stats.morans_i
    """
    n = annot.shape[0]
    annot_demean = annot - np.mean(annot)
    if use_sampvar:
        lower = n - 1
    else:
        lower = n
    m_2 = np.dot(annot_demean, annot_demean) / lower
    right = np.squeeze(weight @ annot_demean[:, None])  # np.dot(weight, annot)
    return annot_demean * right / m_2
