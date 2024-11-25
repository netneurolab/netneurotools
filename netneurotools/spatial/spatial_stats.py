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
    lower = np.sum(annot_demean**2)
    return n * upper / (lower * W)


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


if has_numba:
    _morans_i_numba = njit(_morans_i_numba)


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
        w <- mat2listw(m)
        moran(v, w, 100, Szero(w))
        # or
        moran.test(x, w)

    See Also
    --------
    netneurotools.spatial.spatial_stats.local_morans_i
    """
    if use_numba:
        if not has_numba:
            raise ValueError("Numba not installed; cannot use numba for calculation")
        return _morans_i_numba(annot, weight)
    else:
        return _morans_i_vectorized(annot, weight)


def local_morans_i(annot, weight, use_sampvar=True):
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
        Whether to use sample variance (n - 1) in calculation. Default: True.

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
            localmoran(x, mat2listw(m), mlvar=TRUE)

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


def _gearys_c_vectorized(annot, weight):
    n = annot.shape[0]
    annot_demean = annot - np.mean(annot)
    W = np.sum(weight)
    upper = np.sum(weight * (annot[:, np.newaxis] - annot[np.newaxis, :]) ** 2)
    lower = np.sum(annot_demean**2)
    return (n - 1) * upper / (2 * W * lower)


def _gearys_c_numba(annot, weight):
    n = annot.shape[0]
    annot_demean = annot - np.mean(annot)  # not all occurances are demean-ed
    upper, lower, W = 0.0, 0.0, 0.0
    for i in range(n):
        lower += annot_demean[i] ** 2
        for j in range(n):
            upper += weight[i, j] * (annot[i] - annot[j]) ** 2
            W += weight[i, j]
    return (n - 1) * upper / lower / (2 * W)


if has_numba:
    _gearys_c_numba = njit(_gearys_c_numba)


def gearys_c(annot, weight, use_numba=has_numba):
    r"""
    Calculate Geary's C for spatial autocorrelation.

    Parameters
    ----------
    annot : array-like, shape (n,)
        Array of annotations to calculate Geary's C for.
    weight : array-like, shape (n, n)
        Spatial weight matrix. Note that we do not explicitly check for symmetry
        in the weight matrix, nor zero-diagonal elements.
    use_numba : bool, optional
        Whether to use numba for calculation. Default: True (if numba is
        installed).

    Returns
    -------
    gearys_c : float
        Geary's C value for the given annotations and weight matrix.

    Notes
    -----
    Geary's C is calculated as:

    .. math::
        C = \frac{(n-1)}{2\sum_{i=1}^n \sum_{j=1}^n w_{ij}}
        \frac{\sum_{i=1}^n \sum_{j=1}^n w_{ij}(x_i - x_j)^2}
        {\sum_{i=1}^n(x_i - \bar{x})^2}

    where :math:`n` is the number of observations, :math:`w_{ij}` is the spatial
    weight between observations :math:`i` and :math:`j`, :math:`x_i` is the
    annotation for observation :math:`i`, and :math:`\bar{x}` is the mean
    annotation value.

    The value can be tested using the R pacakge ``spdep``:

    .. code:: R

        x <- rnorm(100)
        m <- matrix(runif(100*100), nrow=100)
        w <- mat2listw(m)
        geary(x, w, 100, 100-1, Szero(w))
        # or
        geary.test(x, w)

    """
    if use_numba:
        if not has_numba:
            raise ValueError("Numba not installed; cannot use numba for calculation")
        return _gearys_c_numba(annot, weight)
    else:
        return _gearys_c_vectorized(annot, weight)


def local_gearys_c(annot, weight, use_sampvar=True):
    r"""
    Calculate local Geary's C for spatial autocorrelation.

    Parameters
    ----------
    annot : array-like, shape (n,)
        Array of annotations to calculate Geary's C for.
    weight : array-like, shape (n, n)
        Spatial weight matrix. Note that we do not explicitly check for symmetry
        in the weight matrix, nor zero-diagonal elements.
    use_sampvar : bool, optional
        Whether to use sample variance (n - 1) in calculation. Default: True.

    Returns
    -------
    local_gearys_c : array, shape (n,)
        Local Geary's C values for the given annotations and weight matrix.

    Notes
    -----
    Local Geary's C is calculated as:

    .. math::
        C_i = \frac{w_{ij} (x_i - x_j)^2}{\sum_{j=1}^n w_{ij} (x_i - x_j)^2}

    where :math:`n` is the number of observations, :math:`w_{ij}` is the spatial
    weight between observations :math:`i` and :math:`j`, :math:`x_i` is the
    annotation for observation :math:`i`, and :math:`\bar{x}` is the mean
    annotation value.

    The value can be tested using the R pacakge ``spdep``:

    .. code:: R

        x <- rnorm(100)
        m <- matrix(runif(100*100), nrow=100)
        localC(v, mat2listw(m))
    """
    n = annot.shape[0]
    annot_demean = annot - np.mean(annot)
    if use_sampvar:
        lower = n - 1
    else:
        lower = n
    m_2 = np.dot(annot_demean, annot_demean) / lower  # unbiased
    annot_diffsq = (annot[:, None] - annot[None, :]) ** 2
    right = np.sum(weight * annot_diffsq, axis=1)
    return right / m_2


def _lees_i_vectorized(annot_1, annot_2, weight):
    n = annot_1.shape[0]
    annot_1_demean = annot_1 - np.mean(annot_1)
    annot_2_demean = annot_2 - np.mean(annot_2)

    # Calculate upper term
    w_x1 = weight @ annot_1_demean
    w_x2 = weight @ annot_2_demean
    upper = np.sum(w_x1 * w_x2)

    # Calculate S2 (sum of squared row sums)
    s2 = np.sum(np.sum(weight, axis=1) ** 2)

    # Calculate lower terms
    lower_1 = np.sum(annot_1_demean ** 2)
    lower_2 = np.sum(annot_2_demean ** 2)

    return n / s2 * upper / np.sqrt(lower_1) / np.sqrt(lower_2)


def _lees_i_numba(annot_1, annot_2, weight):
    n = annot_1.shape[0]
    annot_1_demean = annot_1 - np.mean(annot_1)
    annot_2_demean = annot_2 - np.mean(annot_2)

    upper, lower_1, lower_2, s2 = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        upper_1, upper_2 = 0.0, 0.0
        s2_tmp = 0.0
        for j in range(n):
            upper_1 += weight[i, j] * annot_1_demean[j]
            upper_2 += weight[i, j] * annot_2_demean[j]
            s2_tmp += weight[i, j]
        upper += upper_1 * upper_2
        s2 += s2_tmp ** 2
        #
        lower_1 += annot_1_demean[i] ** 2
        lower_2 += annot_2_demean[i] ** 2

    return n / s2 * upper / np.sqrt(lower_1) / np.sqrt(lower_2)


def lees_i(annot_1, annot_2, weight, use_numba=has_numba):
    r"""
    Calculate Lee's I for spatial autocorrelation.

    Parameters
    ----------
    annot_1 : array-like, shape (n,)
        Array of annotations to calculate Lee's I for.
    annot_2 : array-like, shape (n,)
        Array of annotations to calculate Lee's I for.
    weight : array-like, shape (n, n)
        Spatial weight matrix. Note that we do not explicitly check for symmetry
        in the weight matrix, nor zero-diagonal elements.
    use_numba : bool, optional
        Whether to use numba for calculation. Default: True (if numba is
        installed).

    Returns
    -------
    lees_i : float
        Lee's I value for the given annotations and weight matrix.

    Notes
    -----
    Lee's I is calculated as:

    .. math::
        L(x,y) = \frac{n}{\sum_{i=1}^n(\sum_{j=1}^n w_{ij})^2}
        \frac{\sum_{i=1}^n(\sum_{j=1}^n w_{ij}(x_i - \bar{x}))
        (\sum_{j=1}^n w_{ij}(y_i - \bar{y}))}
        {\sqrt{\sum_{i=1}^n(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^n(y_i - \bar{y})^2}}

    where :math:`n` is the number of observations, :math:`w_{ij}` is the spatial
    weight between observations :math:`i` and :math:`j`, :math:`x_i` is the
    annotation for observation :math:`i`, and :math:`\bar{x}` is the mean annotation
    value. :math:`x` and :math:`y` are the annotations for the two variables.

    The value can be tested using the R pacakge ``spdep``:

    .. code:: R

        x <- rnorm(100)
        y <- rnorm(100)
        m <- matrix(runif(100*100), nrow=100)
        lee(x, y, mat2listw(m), 100)
        # or
        lee.test(x, y, mat2listw(m))
    """
    if use_numba:
        if not has_numba:
            raise ValueError("Numba not installed; cannot use numba for calculation")
        return _lees_i_numba(annot_1, annot_2, weight)
    else:
        return _lees_i_vectorized(annot_1, annot_2, weight)


def local_lees_i(annot_1, annot_2, weight):
    r"""
    Calculate local Lee's I for spatial autocorrelation.

    Parameters
    ----------
    annot_1 : array-like, shape (n,)
        Array of annotations to calculate Lee's I for.
    annot_2 : array-like, shape (n,)
        Array of annotations to calculate Lee's I for.
    weight : array-like, shape (n, n)
        Spatial weight matrix. Note that we do not explicitly check for symmetry
        in the weight matrix, nor zero-diagonal elements.

    Returns
    -------
    local_lees_i : array, shape (n,)
        Local Lee's I values for the given annotations and weight matrix.

    Notes
    -----
    Local Lee's I is calculated as:

    .. math::
        L_i(x,y) = \frac{(\sum_{j=1}^n w_{ij}(x_i - \bar{x}))
        (\sum_{j=1}^n w_{ij}(y_i - \bar{y}))}
        {\sqrt{\sum_{i=1}^n(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^n(y_i - \bar{y})^2}}

    where :math:`n` is the number of observations, :math:`w_{ij}` is the spatial
    weight between observations :math:`i` and :math:`j`, :math:`x_i` is the
    annotation for observation :math:`i`, and :math:`\bar{x}` is the mean annotation
    value. :math:`x` and :math:`y` are the annotations for the two variables.

    The value can be tested using the R pacakge ``spdep``:

    .. code:: R

        x <- rnorm(100)
        y <- rnorm(100)
        m <- matrix(runif(100*100), nrow=100)
        lee(x, y, mat2listw(m), 100)
    """
    n = annot_1.shape[0]
    annot_1_demean = annot_1 - np.mean(annot_1)
    annot_2_demean = annot_2 - np.mean(annot_2)
    lower_1 = np.sqrt(np.sum(annot_1_demean ** 2))
    lower_2 = np.sqrt(np.sum(annot_2_demean ** 2))
    upper_1 = np.squeeze(weight @ annot_1_demean[:, None])  # np.dot(weight, annot)
    upper_2 = np.squeeze(weight @ annot_2_demean[:, None])  # np.dot(weight, annot)
    return n * upper_1 * upper_2 / lower_1 / lower_2
