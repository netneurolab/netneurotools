# -*- coding: utf-8 -*-
"""
Functions for calculating network metrics. Uses naming conventions adopted
from the Brain Connectivity Toolbox (https://sites.google.com/site/bctnet/).
"""

import itertools
import numpy as np
from scipy.linalg import expm
from scipy.stats import ttest_ind
from scipy.sparse.csgraph import shortest_path

try:
    from numba import njit
    use_numba = True
except ImportError:
    use_numba = False


def _binarize(W):
    """
    Binarizes a matrix

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


if use_numba:
    _binarize = njit(_binarize)


def degrees_und(W):
    """
    Computes the degree of each node in `W`

    Parameters
    ----------
    W : (N, N) array_like
        Unweighted, undirected connection weight array.
        Weighted array will be binarized prior to calculation.
        Directedness will be ignored (out degree / row sum taken).

    Returns
    -------
    deg : (N,) numpy.ndarray
        Degree of each node in `W`
    """
    return np.sum(_binarize(W), axis=0)


def degrees_dir(W):
    """
    Computes the in degree and out degree of each node in `W`

    Parameters
    ----------
    W : (N, N) array_like
        Unweighted, directed connection weight array.
        Weighted array will be binarized prior to calculation.

    Returns
    -------
    deg_in : (N,) numpy.ndarray
        In-degree (column sum) of each node in `W`
    deg_out : (N,) numpy.ndarray
        Out-degree (row sum) of each node in `W`
    deg : (N,) numpy.ndarray
        Degree (in-degree + out-degree) of each node in `W`
    """
    W_bin = _binarize(W)
    deg_in = np.sum(W_bin, axis=0)
    deg_out = np.sum(W_bin, axis=1)
    deg = deg_in + deg_out
    return deg_in, deg_out, deg


def distance_wei_floyd(D):
    """
    Computes the shortest path length between all pairs of nodes using
    Floyd-Warshall algorithm.

    Parameters
    ----------
    D : (N, N) array_like
        Connection length or distance matrix.
        Please do the weight-to-distance beforehand.

    Returns
    -------
    spl_mat : (N, N) array_like
        Shortest path length matrix
    p_mat : (N, N) array_like
        Predecessor matrix returned from `scipy.sparse.csgraph.shortest_path`

    Notes
    -----
    This function is a wrapper for `scipy.sparse.csgraph.shortest_path`.
    There may be more than one shortest path between two nodes, and we
    only return the first one found by the algorithm.

    References
    ----------
    .. [1] Floyd, R. W. (1962). Algorithm 97: shortest path. Communications of
       the ACM, 5(6), 345.
    .. [2] Roy, B. (1959). Transitivite et connexite. Comptes Rendus
       Hebdomadaires Des Seances De L Academie Des Sciences, 249(2), 216-218.
    .. [3] Warshall, S. (1962). A theorem on boolean matrices. Journal of the
       ACM (JACM), 9(1), 11-12.
    .. [4] https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm

    See Also
    --------
    netneurotools.metrics.retrieve_shortest_paths
    """
    spl_mat, p_mat = shortest_path(
        D, method="FW", directed=False, return_predecessors=True,
        unweighted=False, overwrite=False
    )
    return spl_mat, p_mat


def retrieve_shortest_paths(s, t, p_mat):
    """
    Returns the shortest paths between two nodes.

    Parameters
    ----------
    s : int
        Source node
    t : int
        Target node
    p_mat : (N, N) array_like
        Predecessor matrix returned from `distance_wei_floyd`

    Returns
    -------
    path : list of int
        List of nodes in the shortest path from `s` to `t`. If no path
        exists, returns `[-1]`.

    See Also
    --------
    netneurotools.metrics.distance_wei_floyd
    """
    if p_mat[s, t] == -9999:
        return [-1]
    path = [t]
    while path[-1] != s:
        t = p_mat[s, t]
        path.append(t)
    return path[::-1]


if use_numba:
    retrieve_shortest_paths = njit(retrieve_shortest_paths)


def communicability_bin(adjacency, normalize=False):
    """
    Computes the communicability of pairs of nodes in `adjacency`

    Parameters
    ----------
    adjacency : (N, N) array_like
        Unweighted, direct/undirected connection weight/length array
    normalize : bool, optional
        Whether to normalize `adjacency` by largest eigenvalue prior to
        calculation of communicability metric. Default: False

    Returns
    -------
    comm : (N, N) numpy.ndarray
        Symmetric array representing communicability of nodes {i, j}

    References
    ----------
    Estrada, E., & Hatano, N. (2008). Communicability in complex networks.
    Physical Review E, 77(3), 036111.

    Examples
    --------
    >>> from netneurotools import metrics

    >>> A = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1]])
    >>> Q = metrics.communicability_bin(A)
    >>> Q
    array([[4.19452805, 0.        , 3.19452805],
           [1.47624622, 2.71828183, 3.19452805],
           [3.19452805, 0.        , 4.19452805]])
    """

    if not np.any(np.logical_or(adjacency == 0, adjacency == 1)):
        raise ValueError('Provided adjancecy matrix must be unweighted.')

    # normalize by largest eigenvalue to prevent communicability metric from
    # "blowing up"
    if normalize:
        norm = np.linalg.eigvals(adjacency).max()
        adjacency = adjacency / norm

    return expm(adjacency)


def communicability_wei(adjacency):
    """
    Computes the communicability of pairs of nodes in `adjacency`

    Parameters
    ----------
    adjacency : (N, N) array_like
        Weighted, direct/undirected connection weight/length array

    Returns
    -------
    cmc : (N, N) numpy.ndarray
        Symmetric array representing communicability of nodes {i, j}

    References
    ----------
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure
    applied to complex brain networks. Journal of the Royal Society Interface,
    6(33), 411-414.

    Examples
    --------
    >>> from netneurotools import metrics

    >>> A = np.array([[2, 0, 3], [0, 2, 1], [0.5, 0, 1]])
    >>> Q = metrics.communicability_wei(A)
    >>> Q
    array([[0.        , 0.        , 1.93581903],
           [0.07810379, 0.        , 0.94712177],
           [0.32263651, 0.        , 0.        ]])
    """

    # negative square root of nodal degrees
    row_sum = adjacency.sum(1)
    neg_sqrt = np.power(row_sum, -0.5)
    square_sqrt = np.diag(neg_sqrt)

    # normalize input matrix
    for_expm = square_sqrt @ adjacency @ square_sqrt

    # calculate matrix exponential of normalized matrix
    cmc = expm(for_expm)
    cmc[np.diag_indices_from(cmc)] = 0

    return cmc


def rich_feeder_peripheral(x, sc, stat='median'):
    """
    Calculates connectivity values in rich, feeder, and peripheral edges.

    Parameters
    ----------
    x : (N, N) numpy.ndarray
        Symmetric correlation or connectivity matrix
    sc : (N, N) numpy.ndarray
        Binary structural connectivity matrix
    stat : {'mean', 'median'}, optional
        Statistic to use over rich/feeder/peripheral links. Default: 'median'

    Returns
    -------
    rfp : (3, k) numpy.ndarray
        Array of median rich (0), feeder (1), and peripheral (2)
        values, defined by `x`. `k` is the maximum degree defined on `sc`.
    pvals : (3, k) numpy.ndarray
        p-value for each link, computed using Welch's t-test.
        Rich links are compared against non-rich links. Feeder links are
        compared against peripheral links. Peripheral links are compared
        against feeder links. T-test is one-sided.

    Notes
    -----
    This code was written by Justine Hansen who promises to fix and even
    optimize the code should any issues arise, provided you let her know.
    """

    stats = ['mean', 'median']
    if stat not in stats:
        raise ValueError(f'Provided stat {stat} not valid.\
                         Must be one of {stats}')

    nnodes = len(sc)
    mask = np.triu(np.ones(nnodes), 1) > 0
    node_degree = degrees_und(sc)
    k = np.max(node_degree).astype(np.int64)
    rfp_label = np.zeros((len(sc[mask]), k))

    for degthresh in range(k):  # for each degree threshold
        hub_idx = np.where(node_degree >= degthresh)  # find the hubs
        hub = np.zeros([nnodes, 1])
        hub[hub_idx, :] = 1

        rfp = np.zeros([nnodes, nnodes])      # for each link, define rfp
        for edge1 in range(nnodes):
            for edge2 in range(nnodes):
                if hub[edge1] + hub[edge2] == 2:
                    rfp[edge1, edge2] = 1  # rich
                if hub[edge1] + hub[edge2] == 1:
                    rfp[edge1, edge2] = 2  # feeder
                if hub[edge1] + hub[edge2] == 0:
                    rfp[edge1, edge2] = 3  # peripheral
        rfp_label[:, degthresh] = rfp[mask]

    rfp = np.zeros([3, k])
    pvals = np.zeros([3, k])
    for degthresh in range(k):

        redfunc = np.median if stat == 'median' else np.mean
        for linktype in range(3):
            rfp[linktype, degthresh] = redfunc(x[mask][rfp_label[:, degthresh]
                                                       == linktype + 1])

        # p-value (one-sided Welch's t-test)
        _, pvals[0, degthresh] = ttest_ind(
            x[mask][rfp_label[:, degthresh] == 1],
            x[mask][rfp_label[:, degthresh] != 1],
            equal_var=False, alternative='greater')
        _, pvals[1, degthresh] = ttest_ind(
            x[mask][rfp_label[:, degthresh] == 2],
            x[mask][rfp_label[:, degthresh] == 3],
            equal_var=False, alternative='greater')
        _, pvals[2, degthresh] = ttest_ind(
            x[mask][rfp_label[:, degthresh] == 3],
            x[mask][rfp_label[:, degthresh] == 2],
            equal_var=False, alternative='greater')

    return rfp, pvals


def navigation_wu(nav_dist_mat, sc_mat):
    """
    Computes network navigation

    Parameters
    ----------
    nav_dist_mat : (N, N) array_like
        Connection length/distance matrix.
    sc_mat : (N, N) array_like
        Structural connectivity matrix, only used to get connectedness.

    Returns
    -------
    nav_sr : float
        Overall navigation success rate
    nav_sr_node : list of float
        Nodal navigation success rate
    nav_path_len : (N, N) array_like
        Navigation path length matrix, infinite if no path exists.
    nav_path_hop : (N, N) array_like
        Navigation path hop matrix, infinite if no path exists.
    nav_paths : list
        List of tuples containing source node, target node, path length,
        path hops, and the full path.

    References
    ----------
    Seguin, C., Van Den Heuvel, M. P., & Zalesky, A. (2018). Navigation
    of brain networks. Proceedings of the National Academy of Sciences,
    115(24), 6297-6302.

    Notes
    -----
    Euclidean distance between nodes are usually used for `nav_dist_mat`.
    Distances returned from this function are also calculated from
    `nav_dist_mat`.
    Use :meth:`netneurotools.metrics.get_navigation_path_length`
    to get path length in other metrics.

    See Also
    --------
    netneurotools.metrics.get_navigation_path_length
    """

    nav_paths = []  # (source, target, distance, hops, path)
    # navigate to the node that is closest to target
    for src in range(len(nav_dist_mat)):
        for tar in range(len(nav_dist_mat)):
            curr_pos = src
            curr_path = [src]
            curr_dist = 0
            while curr_pos != tar:
                neig = np.where(sc_mat[curr_pos, :] != 0)[0]
                if len(neig) == 0:  # not connected
                    curr_path = []
                    curr_dist = np.inf
                    break
                neig_dist_to_tar = nav_dist_mat[neig, tar]
                min_dist_idx = np.argmin(neig_dist_to_tar)

                new_pos = neig[min_dist_idx]
                # Assume it is connected, and only testing for loops.
                # if isempty(next_node)
                # || next_node == last_node
                # || pl_bin > max_hops
                if (new_pos in curr_path):
                    curr_path = []
                    curr_dist = np.inf
                    break
                else:
                    curr_path.append(new_pos)
                    curr_dist += nav_dist_mat[curr_pos, new_pos]
                    curr_pos = new_pos
            nav_paths.append(
                (src, tar, curr_dist, len(curr_path) - 1, curr_path))

    nav_sr = len([_ for _ in nav_paths if _[3] != -1]) / len(nav_paths)

    nav_sr_node = []
    for k, g in itertools.groupby(
        sorted(nav_paths, key=lambda x: x[0]), key=lambda x: x[0]
    ):
        curr_path = list(g)
        nav_sr_node.append(
            len([_ for _ in curr_path if _[3] != -1]) / len(curr_path))

    nav_path_len = np.zeros_like(nav_dist_mat)
    nav_path_hop = np.zeros_like(nav_dist_mat)
    for nav_item in nav_paths:
        i, j, length, hop, _ = nav_item
        if hop != -1:
            nav_path_len[i, j] = length
            nav_path_hop[i, j] = hop
        else:
            nav_path_len[i, j] = np.inf
            nav_path_hop[i, j] = np.inf

    return nav_sr, nav_sr_node, nav_path_len, nav_path_hop, nav_paths


def get_navigation_path_length(nav_paths, alt_dist_mat):
    """
    Get navigation path length from navigation results

    Parameters
    ----------
    nav_paths : list
        Return from netneurotools.metrics.navigation_wu
    alt_dist_mat : (N, N) array_like
        Alternative distance matrix, e.g. geodesic distance.

    Returns
    -------
    nav_path_len : (N, N) array_like
        Navigation path length matrix, in the alternative distance metric.

    Notes
    -----
    Following the original BCT function.
    `pl_wei = get_navigation_path_length(nav_paths, L)`
    L is strength-to-length remapping of the connection weight matrix.
    `pl_dis = get_navigation_path_length(nav_paths, D)`
    D is Euclidean distance between node centroids.

    See Also
    --------
    netneurotools.metrics.navigation_wu
    """

    nav_path_len = np.zeros_like(alt_dist_mat)
    for nav_item in nav_paths:
        i, j, _, hop, path = nav_item
        if hop != -1:
            nav_path_len[i, j] = np.sum(
                [alt_dist_mat[path[_], path[_ + 1]] for _ in range(hop)]
            )
        else:
            nav_path_len[i, j] = np.inf
    return nav_path_len
