"""Functions for supporting network metrics."""

import numpy as np
from .. import has_numba
if has_numba:
    from numba import njit


def _fast_binarize(W):
    """
    Binarize a matrix.

    Parameters
    ----------
    W : (N, N) array_like
        Matrix to be binarized

    Returns
    -------
    binarized : (N, N) numpy.ndarray
        Binarized matrix
    """
    return (W > 0) * 1


if has_numba:
    _fast_binarize = njit(_fast_binarize)


def _graph_laplacian(W):
    r"""
    Compute the graph Laplacian of a weighted adjacency matrix.

    Graph Laplacian is defined as the degree matrix minus the adjacency
    matrix :math:`L = D - W`, where :math:`D` is the degree matrix and
    is defined as :math:`D_{ii} = \sum_j W_{ij}`.

    The graph Laplacian matrix :math:`L` has the form of

    .. math::
        L = \begin{bmatrix}
            d_1 & -w_{12} & \cdots & -w_{1n} \\
            -w_{21} & d_2 & \cdots & -w_{2n} \\
            \vdots & \vdots & \ddots & \vdots \\
            -w_{n1} & -w_{n2} & \cdots & d_n
        \end{bmatrix}

    Parameters
    ----------
    W : (N, N) array_like
        Weighted, directed/undirected connection weight/length array

    Returns
    -------
    L : (N, N) numpy.ndarray
        Graph Laplacian of `W`
    """
    D = np.diag(np.sum(W, axis=0))
    return D - W


if has_numba:
    _graph_laplacian = njit(_graph_laplacian)  # ("float64[:,::1](float64[:,::1])")
