"""Functions for calculating statistical network metrics."""

import numpy as np

from .. import has_numba
if has_numba:
    from numba import njit

from .metrics_utils import _graph_laplacian


def _network_pearsonr_vectorized(annot1, annot2, weight):
    annot1 = annot1 - np.mean(annot1)
    annot2 = annot2 - np.mean(annot2)
    upper = np.sum(np.multiply(weight, np.outer(annot1, annot2)))
    lower1 = np.sum(np.multiply(weight, np.outer(annot1, annot1)))
    lower2 = np.sum(np.multiply(weight, np.outer(annot2, annot2)))
    return upper / np.sqrt(lower1) / np.sqrt(lower2)


def _network_pearsonr_numba(annot1, annot2, weight):
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


if has_numba:
    _network_pearsonr_numba = njit(_network_pearsonr_numba)


def network_pearsonr(annot1, annot2, weight, use_numba=has_numba):
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
    use_numba : bool, optional
        Whether to use numba for calculation. Default: True
        (if numba is available).

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
    if use_numba:
        if not has_numba:
            raise ValueError("Numba not installed; cannot use numba for calculation")
        return _network_pearsonr_numba(annot1, annot2, weight)
    else:
        return _network_pearsonr_vectorized(annot1, annot2, weight)


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


if has_numba:
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


if has_numba:
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
    if has_numba:
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


if has_numba:
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
    if has_numba:
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


if has_numba:
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


if has_numba:
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
    if has_numba:
        polariz_sq = _quadratic_form(Q_star, diff, diff, squared=False)
    else:
        polariz_sq = (diff.T @ Q_star @ diff)
    return np.sqrt(polariz_sq)


def _network_variance_vectorized(vec, D):
    p = vec / np.sum(vec)
    return 0.5 * (p.T @ np.multiply(D, D) @ p)


def _network_variance_numba(vec, D):
    p = vec / np.sum(vec)
    return 0.5 * _quadratic_form(D, p, p, squared=True)


if has_numba:
    _network_variance_numba = njit(_network_variance_numba)


def network_variance(vec, D, use_numba=has_numba):
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
    use_numba : bool, optional
        Whether to use numba for calculation. Default: True
        (if numba is available).

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
    if use_numba:
        if not has_numba:
            raise ValueError("Numba not installed; cannot use numba for calculation")
        return _network_variance_numba(vec, D)
    else:
        return _network_variance_vectorized(vec, D)


def _network_covariance_vectorized(joint_pmat, D, calc_marginal=True):
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


def _network_covariance_numba(joint_pmat, D, calc_marginal=True):
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


if has_numba:
    _network_covariance_numba = njit(_network_covariance_numba)


def network_covariance(joint_pmat, D, calc_marginal=True, use_numba=has_numba):
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
    use_numba : bool, optional
        Whether to use numba for calculation. Default: True
        (if numba is available).

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
    if use_numba:
        if not has_numba:
            raise ValueError("Numba not installed; cannot use numba for calculation")
        return _network_covariance_numba(joint_pmat, D, calc_marginal)
    else:
        return _network_covariance_vectorized(joint_pmat, D, calc_marginal)
