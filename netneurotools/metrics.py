# -*- coding: utf-8 -*-
"""
Functions for calculating network metrics.

Uses naming conventions adopted from the Brain Connectivity
Toolbox (https://sites.google.com/site/bctnet/).
"""

import itertools
import numpy as np
import scipy
from scipy.stats import ttest_ind
from scipy.sparse.csgraph import shortest_path

try:
    from numba import njit
    use_numba = True
except ImportError:
    use_numba = False


def _binarize(W):
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


if use_numba:
    _binarize = njit(_binarize)


def degrees_und(W):
    """
    Compute the degree of each node in `W`.

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
    Compute the in degree and out degree of each node in `W`.

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
    Compute the all-pairs shortest path length using Floyd-Warshall algorithm.

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
    netneurotools.metrics.retrieve_shortest_path
    """
    spl_mat, p_mat = shortest_path(
        D, method="FW", directed=True, return_predecessors=True,
        unweighted=False, overwrite=False
    )
    return spl_mat, p_mat


def retrieve_shortest_path(s, t, p_mat):
    """
    Return the shortest paths between two nodes.

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
    retrieve_shortest_path = njit(retrieve_shortest_path)


def communicability_bin(adjacency, normalize=False):
    """
    Compute the communicability of pairs of nodes in `adjacency`.

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

    return scipy.sparse.linalg.expm(adjacency)


def communicability_wei(adjacency):
    """
    Compute the communicability of pairs of nodes in `adjacency`.

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
    cmc = scipy.sparse.linalg.expm(for_expm)
    cmc[np.diag_indices_from(cmc)] = 0

    return cmc


def rich_feeder_peripheral(x, sc, stat='median'):
    """
    Calculate connectivity values in rich, feeder, and peripheral edges.

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
    Compute network navigation.

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
    for _, g in itertools.groupby(
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
    Get navigation path length from navigation results.

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


def search_information(W, D, has_memory=False):
    """
    Calculate search information.

    This function implements search information, computes the amount
    of information (measured in bits) that a random walker needs to
    follow the shortest path between a given pair of nodes.

    This function is adapted and optimized from the Brain Connectivity Toolbox.

    .. warning::
       Test before use.

    Parameters
    ----------
    W : (N, N) ndarray
        Weighted/unweighted, directed/undirected connection weight matrix.
    D : (N, N) ndarray
        Weighted/unweighted, directed/undirected connection length or
        distance matrix. Please do the weight-to-distance beforehand.
    has_memory : bool, optional
        Memory for random walker, Default: False

    Returns
    -------
    SI : (N, N) ndarray
        Pairwise search information matrix. The diagonal is set to NaN.
        Edges without a valid shortest path are set to np.inf.
        It is not guaranteed to be symmetric even if input is symmetric.

    References
    ----------
    .. [1] Rosvall, M., Trusina, A., Minnhagen, P., & Sneppen, K. (2005).
       Networks and cities: An information perspective. Physical Review
       Letters, 94(2), 028701.
    .. [2] Goñi, J., Van Den Heuvel, M. P., Avena-Koenigsberger,
       A., Velez de Mendizabal, N., Betzel, R. F., Griffa, A., ... &
       Sporns, O. (2014). Resting-brain functional connectivity predicted
       by analytic measures of network communication. Proceedings of the
       National Academy of Sciences, 111(2), 833-838.
    """
    N = len(W)

    is_sym = True if np.allclose(W, W.T) else False

    T = W / np.sum(W, axis=1)[:, None]
    _, p_mat = distance_wei_floyd(D)

    SI = np.zeros((N, N))

    if is_sym:  # symmetric case
        for i in range(N):
            for j in range(i + 1, N):
                path = retrieve_shortest_path(i, j, p_mat)
                if path[0] != -1:  # no path, depends on retrieve_shortest_path
                    if has_memory:
                        pr_step_ff = \
                            [0] + [T[i, j] for i, j in zip(path[:-1], path[1:])]
                        pr_step_bk = \
                            [T[i, j] for i, j in zip(path[1:], path[:-1])] + [0]
                        pr_step_ff = [
                            i / (1 - j)
                            for i, j in zip(pr_step_ff[1:], pr_step_ff[:-1])
                        ]
                        pr_step_bk = [
                            i / (1 - j)
                            for i, j in zip(pr_step_bk[:-1], pr_step_bk[1:])
                        ]
                    else:
                        pr_step_ff = \
                            [T[i, j] for i, j in zip(path[:-1], path[1:])]
                        pr_step_bk = \
                            [T[i, j] for i, j in zip(path[1:], path[:-1])]
                    SI[i, j] = -np.log2(np.prod(pr_step_ff))
                    SI[j, i] = -np.log2(np.prod(pr_step_bk))
                else:
                    SI[i, j] = np.inf
                    SI[j, i] = np.inf
    else:  # asymmetric case
        for i in range(N):
            for j in range(N):
                if i == j:  # skip self connection
                    continue
                path = retrieve_shortest_path(i, j, p_mat)
                if path[0] != -1:  # no path, depends on retrieve_shortest_path
                    if has_memory:
                        pr_step_ff = \
                            [0] + [T[i, j] for i, j in zip(path[:-1], path[1:])]
                        pr_step_ff = [
                            i / (1 - j)
                            for i, j in zip(pr_step_ff[1:], pr_step_ff[:-1])
                        ]
                    else:
                        pr_step_ff = [T[i, j] for i, j in zip(path[:-1], path[1:])]
                    SI[i, j] = -np.log2(np.prod(pr_step_ff))
                else:
                    SI[i, j] = np.inf

    np.fill_diagonal(SI, np.nan)

    return SI


def path_transitivity(D):
    """
    Calculate path transitivity.

    This function implements path transitivity, calculating the density of
    local detours (triangles) that are available along the shortest paths
    between all pairs of nodes.

    This function is adapted and optimized from the Brain Connectivity Toolbox.

    .. warning::
       Test before use.

    Parameters
    ----------
    D : (N, N) ndarray
        Weight or connection length matrix. Length matrix is recommended and
        transform should have been applied.

    Returns
    -------
    T_mat : (N, N) ndarray
        Path transitivity matrix

    References
    ----------
    .. [1] Goñi, J., Van Den Heuvel, M. P., Avena-Koenigsberger,
       A., Velez de Mendizabal, N., Betzel, R. F., Griffa, A., ... &
       Sporns, O. (2014). Resting-brain functional connectivity predicted
       by analytic measures of network communication. Proceedings of the
       National Academy of Sciences, 111(2), 833-838.
    """
    n = len(D)
    m = np.zeros((n, n))
    T_mat = np.zeros((n, n))

    deg_wu = np.sum(D, axis=0)

    for i in range(n - 1):
        for j in range(i + 1, n):
            sig_and = np.logical_and(D[i, :], D[j, :])
            m[i, j] = np.dot(D[i, :] + D[j, :], sig_and) \
                / (deg_wu[i] + deg_wu[j] - 2 * D[i, j])
    m += m.transpose()

    _, p_mat = distance_wei_floyd(D)

    for i in range(n - 1):
        for j in range(i + 1, n):
            path = retrieve_shortest_path(i, j, p_mat)
            K = len(path)
            T_mat[i, j] = 2 \
                * sum([m[i, j] for i, j in itertools.combinations(path, 2)]) \
                / (K * (K - 1))
    T_mat += T_mat.transpose()

    return T_mat


def flow_graph(W, r=None, t=1):
    """
    Calculate flow graph.

    This function implements flow graph, instantiates a continuous
    time random walk on network. Waiting time for walkers at each
    node are distributed as Poisson with rate parameter r.
    This function returns the flow graph at time t.

    .. warning::
       Test before use.

    Parameters
    ----------
    W : (N, N) ndarray
        Symmetric adjacency matrix.
    r : (N,) or (N, 1) ndarray, optional
        Rate parameter. Will be set to np.ones((N, 1)) if not specified.
        Default: None
    t : int, optional
        Markov time. Default: 1

    Returns
    -------
    dyn : (N, N) ndarray
        flow graph at time T

    References
    ----------
    .. [1] Lambiotte, R., Sinatra, R., Delvenne, J. C., Evans, T. S.,
       Barahona, M., & Latora, V. (2011). Flow graphs: Interweaving
       dynamics and structure. Physical Review E, 84(1), 017102.
    .. [2] https://github.com/brain-networks/local_scfc/blob/main/fcn/fcn_flow_graph.m
    """
    if r is None:
        r = np.ones((W.shape[0], 1))
    else:
        if r.ndim == 1:
            r = r[:, None]
    deg_wu = np.sum(W, axis=0, keepdims=True)  # (1, N)
    deg_rate = np.sum(deg_wu / r, axis=0, keepdims=True)  # (N, N) => (1, N)
    ps = deg_wu / (deg_rate * r)  # (1, N) / (N, N) => (N, N)
    laplacian = np.diagflat(r) - np.multiply(np.divide(W, deg_wu), r)  # elementwise
    dyn = np.multiply(
        deg_rate * scipy.sparse.linalg.expm(-t * laplacian),
        ps
    )  # elementwise
    dyn = (dyn + dyn.T) / 2
    return dyn


def mean_first_passage_time(W, tol=1e-3):
    """
    Calculate mean first passage time.

    The first passage time from i to j is the expected number of steps it takes
    a random walker starting at node i to arrive for the first time at node j.
    The mean first passage time is not a symmetric measure: `mfpt(i,j)` may be
    different from `mfpt(j,i)`.

    This function is adapted and optimized from the Brain Connectivity Toolbox.

    .. warning::
       Test before use.

    Parameters
    ----------
    W : (N x N) ndarray
        Weighted/unweighted, direct/undirected connection weight/length array
    tol : float, optional
        Tolerance for eigenvalue of 1. Default: 1e-3

    Returns
    -------
    mfpt : (N x N) ndarray
        Pairwise mean first passage time array

    References
    ----------
    .. [1] Goñi, J., Avena-Koenigsberger, A., Velez de Mendizabal, N.,
       van den Heuvel, M. P., Betzel, R. F., & Sporns, O. (2013). Exploring the
       morphospace of communication efficiency in complex networks. PLoS One,
       8(3), e58070.
    """
    P = W / np.sum(W, axis=1)[:, None]  # transition matrix
    n = len(P)
    D, V = np.linalg.eig(P.T)
    D_minidx = np.argmin(np.abs(D - 1))

    if D[D_minidx] > 1 + tol:
        raise ValueError(
            f"Cannot find eigenvalue of 1. Minimum eigenvalue is larger than {tol}."
        )

    w = V[:, D_minidx][None, :]
    w /= np.sum(w)
    W_prob = np.real(np.repeat(w, n, 0))
    Z = np.linalg.inv(np.eye(n) - P + W_prob)  # fundamental matrix
    mfpt = (np.repeat(np.diag(Z)[None, :], n, 0) - Z) / W_prob
    return mfpt


def diffusion_efficiency(W):
    """
    Calculate diffusion efficiency.

    The diffusion efficiency between nodes i and j is the inverse of the
    mean first passage time from i to j, that is the expected number of
    steps it takes a random walker starting at node i to arrive for the
    first time at node j. Note that the mean first passage time is not a
    symmetric measure -- mfpt(i,j) may be different from mfpt(j,i) -- and
    the pair-wise diffusion efficiency matrix is hence also not symmetric.

    This function is adapted and optimized from the Brain Connectivity Toolbox.

    .. warning::
       Test before use.

    Parameters
    ----------
    W : (N x N) ndarray
        Weighted/unweighted, direct/undirected connection weight/length array

    Returns
    -------
    GE_diff : float
        Global diffusion efficiency
    E_diff : (N x N) ndarray
        Pair-wise diffusion efficiency array

    References
    ----------
    .. [1] Goñi, J., Avena-Koenigsberger, A., Velez de Mendizabal, N.,
       van den Heuvel, M. P., Betzel, R. F., & Sporns, O. (2013). Exploring the
       morphospace of communication efficiency in complex networks. PLoS One,
       8(3), e58070.
    """
    n = W.shape[0]
    mfpt = mean_first_passage_time(W)
    E_diff = np.divide(1, mfpt)
    np.fill_diagonal(E_diff, 0.0)
    GE_diff = np.sum(E_diff) / (n * (n - 1))
    return GE_diff, E_diff


def resource_efficiency_bin(W_bin, lambda_prob=0.5):
    """
    Calculate resource efficiency and shortest-path probability.

    The resource efficiency between nodes i and j is inversly proportional
    to the amount of resources (i.e. number of particles or messages)
    required to ensure with probability 0 < lambda < 1 that at least one of
    them will arrive at node j in exactly SPL steps, where SPL is the
    length of the shortest-path between i and j.

    The shortest-path probability between nodes i and j is the probability
    that a single random walker starting at node i will arrive at node j by
    following (one of) the shortest path(s).

    This function is adapted and optimized from the Brain Connectivity Toolbox.

    .. warning::
       Test before use.

    Parameters
    ----------
    W : (N x N) array_like
        Binary (unweighted) undirected connection matrix.
    lambda_prob : float, optional
        Probability of reaching the target node. Default: 0.5

    Returns
    -------
    E_res : (N x N) ndarray
        Resource efficiency array
    prob_spl : (N x N) ndarray
        Shortest-path probability array

    References
    ----------
    .. [1] Goñi, J., Avena-Koenigsberger, A., Velez de Mendizabal, N.,
       van den Heuvel, M. P., Betzel, R. F., & Sporns, O. (2013). Exploring the
       morphospace of communication efficiency in complex networks. PLoS One,
       8(3), e58070.
    """
    W_bin = _binarize(W_bin)
    if not (0 < lambda_prob < 1):
        raise ValueError("lambda_prob must be between 0 and 1.")

    N = W_bin.shape[0]
    spl_mat, _ = distance_wei_floyd(W_bin)
    spl_mat = spl_mat.astype(int)
    T = W_bin / np.sum(W_bin, axis=1)[:, None]

    L_unique = np.unique(spl_mat)
    prob_spl = np.zeros((N, N))
    z = np.zeros((N, N))

    for L_value in L_unique:
        if L_value == 0:
            continue
        L_locs = (spl_mat == L_value)
        h_cols = np.where(L_locs)[1]
        h_vec = np.unique(h_cols)

        prob_aux = np.zeros((N, N))
        z_aux = np.zeros((N, N))
        for h_value in h_vec:
            B_h = T.copy()
            B_h[h_value, :] = 0
            B_h[h_value, h_value] = 1
            B_h_L = np.linalg.matrix_power(B_h, L_value)
            prob_aux[:, h_value] = B_h_L[:, h_value]
            z_aux[:, h_value] = np.divide(
                np.ones((N,)) * np.log(1 - lambda_prob),
                np.log(1 - B_h_L[:, h_value])
            )

        prob_aux[~L_locs] = 0
        prob_spl += prob_aux

        z_aux[~L_locs] = 0
        z += z_aux

    np.fill_diagonal(prob_spl, 0.0)
    z[prob_spl == 1] = 1
    E_res = 1 / z
    np.fill_diagonal(E_res, 0.0)

    return E_res, prob_spl


def matching_ind_und(W):
    """
    Calculate undirected matching index.

    M0 = MATCHING_IND_UND(CIJ) computes matching index for undirected
    graph specified by adjacency matrix CIJ. Matching index is a measure of
    similarity between two nodes' connectivity profiles (excluding their
    mutual connection, should it exist).

    This function is adapted and optimized from the Brain Connectivity Toolbox.

    .. warning::
       Test before use.

    Parameters
    ----------
    W : (N x N) ndarray
        Undirected connection matrix.

    Returns
    -------
    M0 : (N x N) ndarray
        Matching index matrix

    References
    ----------
    .. [1] Goñi, J., Van Den Heuvel, M. P., Avena-Koenigsberger,
       A., Velez de Mendizabal, N., Betzel, R. F., Griffa, A., ... &
       Sporns, O. (2014). Resting-brain functional connectivity predicted
       by analytic measures of network communication. Proceedings of the
       National Academy of Sciences, 111(2), 833-838.
    """
    n = W.shape[0]
    K = np.sum(W, axis=0)
    R_ind = np.nonzero(K)[0]
    N = len(R_ind)
    CIJ = W[np.ix_(R_ind, R_ind)]
    M = np.zeros((N, N))

    for i in range(N):
        c1 = CIJ[i, :]
        use = np.logical_or(c1 > 0, CIJ > 0)
        use[:, i] = False
        np.fill_diagonal(use, False)

        ncon = np.sum((c1 + CIJ) * use, axis=1)
        ncon_and = np.logical_and(np.logical_and(c1 > 0, CIJ > 0), use)
        ncon_and_sum = np.sum(ncon_and, axis=1)
        M[:, i] = 2 * ncon_and_sum / ncon

    np.fill_diagonal(M, 0)
    M[np.isnan(M)] = 0
    M0 = np.zeros((n, n))
    M0[np.ix_(R_ind, R_ind)] = M
    return M0


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


if use_numba:
    _graph_laplacian = njit(_graph_laplacian)  # ("float64[:,::1](float64[:,::1])")
