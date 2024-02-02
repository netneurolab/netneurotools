# -*- coding: utf-8 -*-
"""Functions for generating group-level networks from individual measurements."""

import numpy as np
from scipy.sparse import csgraph
from sklearn.utils.validation import (check_random_state, check_array,
                                      check_consistent_length)

from . import utils

try:
    from numba import njit
    use_numba = True
except ImportError:
    use_numba = False


def func_consensus(data, n_boot=1000, ci=95, seed=None):
    """
    Calculate thresholded group consensus functional connectivity graph.

    This function concatenates all time series in `data` and computes a group
    correlation matrix based on this extended time series. It then generates
    length `T` bootstrapped samples from the concatenated matrix and estimates
    confidence intervals for all correlations. Correlations whose sign is
    consistent across bootstraps are retained; inconsistent correlations are
    set to zero.

    If `n_boot` is set to 0 or None a simple, group-averaged functional
    connectivity matrix is estimated, instead.

    Parameters
    ----------
    data : (N, T, S) array_like (or a list of S arrays, each shaped as (N, T))
        Pre-processed functional time series, where `N` is the number of nodes,
        `T` is the number of volumes in the time series, and `S` is the number
        of subjects.
    n_boot : int, optional
        Number of bootstraps for which to generate correlation. Default: 1000
    ci : (0, 100) float, optional
        Confidence interval for which to assess the reliability of correlations
        with bootstraps. Default: 95
    seed : int, optional
        Random seed. Default: None

    Returns
    -------
    consensus : (N, N) numpy.ndarray
        Thresholded, group-level correlation matrix

    References
    ----------
    Mišić, B., Betzel, R. F., Nematzadeh, A., Goni, J., Griffa, A., Hagmann,
    P., Flammini, A., Ahn, Y.-Y., & Sporns, O. (2015). Cooperative and
    competitive spreading dynamics on the human connectome. Neuron, 86(6),
    1518-1529.
    """
    # check inputs
    rs = check_random_state(seed)
    if ci > 100 or ci < 0:
        raise ValueError("`ci` must be between 0 and 100.")

    # group-average functional connectivity matrix desired instead of bootstrap
    if n_boot == 0 or n_boot is None:
        if isinstance(data, list):
            corrs = [np.corrcoef(sub) for sub in data]
        else:
            corrs = [np.corrcoef(data[..., sub]) for sub in
                     range(data.shape[-1])]
        return np.nanmean(corrs, axis=0)

    if isinstance(data, list):
        collapsed_data = np.hstack(data)
        nsample = int(collapsed_data.shape[-1] / len(data))
    else:
        collapsed_data = data.reshape((len(data), -1), order='F')
        nsample = data.shape[1]

    consensus = np.corrcoef(collapsed_data)

    # only keep the upper triangle for the bootstraps to save on memory usage
    triu_inds = np.triu_indices_from(consensus, k=1)
    bootstrapped_corrmat = np.zeros((len(triu_inds[0]), n_boot))

    # generate `n_boot` bootstrap correlation matrices by sampling `t` time
    # points from the concatenated time series
    for boot in range(n_boot):
        inds = rs.randint(collapsed_data.shape[-1], size=nsample)
        bootstrapped_corrmat[..., boot] = \
            np.corrcoef(collapsed_data[:, inds])[triu_inds]

    # extract the CIs from the bootstrapped correlation matrices
    # we don't need the input anymore so overwrite it
    bootstrapped_ci = np.percentile(bootstrapped_corrmat, [100 - ci, ci],
                                    axis=-1, overwrite_input=True)

    # remove unreliable (i.e., CI zero-crossing) correlations
    # if the signs of the bootstrapped confidence intervals are different
    # (i.e., their signs sum to 0), then we want to remove them
    # so, take the logical not of the CI (CI = 0 ---> True) and create a mask
    # then, set all connections from the consensus array inside the mask to 0
    remove_inds = np.logical_not(np.sign(bootstrapped_ci).sum(axis=0))
    mask = np.zeros_like(consensus, dtype=bool)
    mask[triu_inds] = remove_inds
    consensus[mask + mask.T] = 0

    return consensus


def _ecdf(data):
    """
    Estimate empirical cumulative distribution function of `data`.

    Taken directly from StackOverflow. See original answer at
    https://stackoverflow.com/questions/33345780.

    Parameters
    ----------
    data : array_like

    Returns
    -------
    prob : numpy.ndarray
        Cumulative probability
    quantiles : numpy.darray
        Quantiles
    """
    sample = np.atleast_1d(data)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    prob = np.cumsum(counts).astype(float) / sample.size

    # match MATLAB
    prob, quantiles = np.append([0], prob), np.append(quantiles[0], quantiles)

    return prob, quantiles


def struct_consensus(data, distance, hemiid, weighted=False):
    """
    Calculate distance-dependent group consensus structural connectivity graph.

    Takes as input a weighted stack of connectivity matrices with dimensions
    (N, N, S) where `N` is the number of nodes and `S` is the number of
    matrices or subjects. The matrices must be weighted, and ideally with
    continuous weights (e.g. fractional anisotropy rather than streamline
    count). The second input is a pairwise distance matrix, where distance(i,j)
    is the Euclidean distance between nodes i and j. The final input is an
    (N, 1) vector which labels nodes as belonging to the right (`hemiid==0`) or
    left (`hemiid=1`) hemisphere (note that these values can be flipped as long
    as `hemiid` contains only values of 0 and 1).

    This function estimates the average edge length distribution and builds
    a group-averaged connectivity matrix that approximates this distribution
    with density equal to the mean density across subjects.

    The algorithm works as follows:

    1. Estimate the cumulative edge length distribution,
    2. Divide the distribution into M length bins, one for each edge that will
       be added to the group-average matrix, and
    3. Within each bin, select the edge that is most consistently expressed
       expressed across subjects, breaking ties according to average edge
       weight (which is why the input matrix `data` must be weighted).

    The algorithm works separately on within/between hemisphere links.

    Parameters
    ----------
    data : (N, N, S) array_like
        Weighted connectivity matrices (i.e., fractional anisotropy), where `N`
        is nodes and `S` is subjects
    distance : (N, N) array_like
        Array where `distance[i, j]` is the Euclidean distance between nodes
        `i` and `j`
    hemiid : (N, 1) array_like
        Hemisphere designation for `N` nodes where a value of 0/1 indicates
        node `N_{i}` is in the right/left hemisphere, respectively
    weighted : bool
        Flag indicating whether or not to return a weighted consensus map. If
        `True`, the consensus will be multiplied by the mean of `data`.

    Returns
    -------
    consensus : (N, N) numpy.ndarray
        Binary (default) or mean-weighted group-level connectivity matrix

    References
    ----------
    Betzel, R. F., Griffa, A., Hagmann, P., & Mišić, B. (2018). Distance-
    dependent consensus thresholds for generating group-representative
    structural brain networks. Network Neuroscience, 1-22.
    """
    # confirm input shapes are as expected
    check_consistent_length(data, distance, hemiid)
    try:
        hemiid = check_array(hemiid, ensure_2d=True)
    except ValueError:
        raise ValueError('Provided hemiid must be a 2D array. Reshape your '
                         'data using array.reshape(-1, 1) and try again.') from None

    num_node, _, num_sub = data.shape      # info on connectivity matrices
    pos_data = data > 0                    # location of + values in matrix
    pos_data_count = pos_data.sum(axis=2)  # num sub with + values at each node

    with np.errstate(divide='ignore', invalid='ignore'):
        average_weights = data.sum(axis=2) / pos_data_count

    # empty array to hold inter/intra hemispheric connections
    consensus = np.zeros((num_node, num_node, 2))

    for conn_type in range(2):  # iterate through inter/intra hemisphere conn
        if conn_type == 0:      # get inter hemisphere edges
            inter_hemi = (hemiid == 0) @ (hemiid == 1).T
            keep_conn = np.logical_or(inter_hemi, inter_hemi.T)
        else:                   # get intra hemisphere edges
            right_hemi = (hemiid == 0) @ (hemiid == 0).T
            left_hemi = (hemiid == 1) @ (hemiid == 1).T
            keep_conn = np.logical_or(right_hemi @ right_hemi.T,
                                      left_hemi @ left_hemi.T)

        # mask the distance array for only those edges we want to examine
        full_dist_conn = distance * keep_conn
        upper_dist_conn = np.atleast_3d(np.triu(full_dist_conn))

        # generate array of weighted (by distance), positive edges across subs
        pos_dist = pos_data * upper_dist_conn
        pos_dist = pos_dist[np.nonzero(pos_dist)]

        # determine average # of positive edges across subs
        # we will use this to bin the edge weights
        avg_conn_num = len(pos_dist) / num_sub

        # estimate empirical CDF of weighted, positive edges across subs
        cumprob, quantiles = _ecdf(pos_dist)
        cumprob = np.round(cumprob * avg_conn_num).astype(int)

        # empty array to hold group-average matrix for current connection type
        # (i.e., inter/intra hemispheric connections)
        group_conn_type = np.zeros((num_node, num_node))

        # iterate through bins (for edge weights)
        for n in range(1, int(avg_conn_num) + 1):
            # get current quantile of interest
            curr_quant = quantiles[np.logical_and(cumprob >= (n - 1),
                                                  cumprob < n)]
            if curr_quant.size == 0:
                continue

            # find edges in distance connectivity matrix w/i current quantile
            mask = np.logical_and(full_dist_conn >= curr_quant.min(),
                                  full_dist_conn <= curr_quant.max())
            i, j = np.where(np.triu(mask))  # indices of edges of interest

            c = pos_data_count[i, j]   # get num sub with + values at edges
            w = average_weights[i, j]  # get averaged weight of edges

            # find locations of edges most commonly represented across subs
            indmax = np.argwhere(c == c.max())

            # determine index of most frequent edge; break ties with higher
            # weighted edge
            if indmax.size == 1:  # only one edge found
                group_conn_type[i[indmax], j[indmax]] = 1
            else:                 # multiple edges found
                indmax = indmax[np.argmax(w[indmax])]
                group_conn_type[i[indmax], j[indmax]] = 1

        consensus[:, :, conn_type] = group_conn_type

    # collapse across hemispheric connections types and make symmetrical array
    consensus = consensus.sum(axis=2)
    consensus = np.logical_or(consensus, consensus.T).astype(int)

    if weighted:
        consensus = consensus * np.mean(data, axis=2)
    return consensus


def binarize_network(network, retain=10, keep_diag=False):
    """
    Keep top `retain` % of connections in `network` and binarizes.

    Uses the upper triangle for determining connection percentage, which may
    result in disconnected nodes. If this behavior is not desired see
    :py:func:`netneurotools.networks.threshold_network`.

    Parameters
    ----------
    network : (N, N) array_like
        Input graph
    retain : [0, 100] float, optional
        Percent connections to retain. Default: 10
    keep_diag : bool, optional
        Whether to keep the diagonal instead of setting it to 0. Default: False

    Returns
    -------
    binarized : (N, N) numpy.ndarray
        Binarized, thresholded graph

    See Also
    --------
    netneurotools.networks.threshold_network
    """
    if retain < 0 or retain > 100:
        raise ValueError('Value provided for `retain` is outside [0, 100]: {}'
                         .format(retain))

    prctile = 100 - retain
    triu = utils.get_triu(network)
    thresh = np.percentile(triu, prctile, axis=0, keepdims=True)
    binarized = np.array(network > thresh, dtype=int)

    if not keep_diag:
        binarized[np.diag_indices(len(binarized))] = 0

    return binarized


def threshold_network(network, retain=10):
    """
    Keep top `retain` % of connections in `network` and binarizes.

    Uses a minimum spanning tree to ensure that no nodes are disconnected from
    the resulting thresholded graph

    Parameters
    ----------
    network : (N, N) array_like
        Input graph
    retain : [0, 100] float, optional
        Percent connections to retain. Default: 10

    Returns
    -------
    thresholded : (N, N) numpy.ndarray
        Binarized, thresholded graph

    See Also
    --------
    netneurotools.networks.binarize_network
    """
    if retain < 0 or retain > 100:
        raise ValueError('Value provided for `retain` must be a percent '
                         'in range [0, 100]. Provided: {}'.format(retain))

    # get number of nodes in graph and invert weights (MINIMUM spanning tree)
    nodes = len(network)
    graph = np.triu(network * -1)

    # find MST and count # of edges in graph
    mst = csgraph.minimum_spanning_tree(graph).todense()
    mst_edges = np.sum(mst != 0)

    # determine # of remaining edges and ensure we're not over the limit
    remain = int((retain / 100) * ((nodes * (nodes - 1)) / 2)) - mst_edges
    if remain < 0:
        raise ValueError('Minimum spanning tree with {} edges exceeds desired '
                         'connection density of {}% ({} edges). Cannot '
                         'proceed with graph creation.'
                         .format(mst_edges, retain, remain + mst_edges))

    # zero out edges already in MST and then get indices of next best edges
    graph -= mst
    inds = utils.get_triu(graph).argsort()[:remain]
    inds = tuple(e[inds] for e in np.triu_indices_from(graph, k=1))

    # add edges to MST, symmetrize, and convert to binary matrix
    mst[inds] = graph[inds]
    mst = np.array((mst + mst.T) != 0, dtype=int)

    return mst


def match_length_degree_distribution(W, D, nbins=10, nswap=1000,
                                     replacement=False, weighted=True,
                                     seed=None):
    """
    Generate degree- and edge length-preserving surrogate connectomes.

    Parameters
    ----------
    W : (N, N) array-like
        weighted or binary symmetric connectivity matrix.
    D : (N, N) array-like
        symmetric distance matrix.
    nbins : int
        number of distance bins (edge length matrix is performed by swapping
        connections in the same bin). Default = 10.
    nswap : int
        total number of edge swaps to perform. Recommended = nnodes * 20
        Default = 1000.
    replacement : bool, optional
        if True all the edges are available for swapping. Default= False
    weighted : bool, optional
        Whether to return weighted rewired connectivity matrix. Default = True
    seed : float, optional
        Random seed. Default = None

    Returns
    -------
    newB : (N, N) array-like
        binary rewired matrix
    newW: (N, N) array-like
        weighted rewired matrix. Returns matrix of zeros if weighted=False.
    nr : int
        number of successful rewires

    Notes
    -----
    Takes a weighted, symmetric connectivity matrix `data` and Euclidean/fiber
    length matrix `distance` and generates a randomized network with:
        1. exactly the same degree sequence
        2. approximately the same edge length distribution
        3. exactly the same edge weight distribution
        4. approximately the same weight-length relationship

    Reference
    ---------
    Betzel, R. F., Bassett, D. S. (2018) Specificity and robustness of
    long-distance connections in weighted, interareal connectomes. PNAS.

    """
    rs = check_random_state(seed)
    N = len(W)
    # divide the distances by lengths
    bins = np.linspace(D[D.nonzero()].min(), D[D.nonzero()].max(), nbins + 1)
    bins[-1] += 1
    L = np.zeros((N, N))
    for n in range(nbins):
        i, j = np.where(np.logical_and(bins[n] <= D, D < bins[n + 1]))
        L[i, j] = n + 1

    # binarized connectivity
    B = (W != 0).astype(np.int_)

    # existing edges (only upper triangular cause it's symmetric)
    cn_x, cn_y = np.where(np.triu((B != 0) * B, k=1))

    tries = 0
    nr = 0
    newB = np.copy(B)

    while ((len(cn_x) >= 2) & (nr < nswap)):
        # choose randomly the edge to be rewired
        r = rs.randint(len(cn_x))
        n_x, n_y = cn_x[r], cn_y[r]
        tries += 1

        # options to rewire with
        # connected nodes that doesn't involve (n_x, n_y)
        index = (cn_x != n_x) & (cn_y != n_y) & (cn_y != n_x) & (cn_x != n_y)
        if len(np.where(index)[0]) == 0:
            cn_x = np.delete(cn_x, r)
            cn_y = np.delete(cn_y, r)

        else:
            ops1_x, ops1_y = cn_x[index], cn_y[index]
            # options that will preserve the distances
            # (ops1_x, ops1_y) such that
            # L(n_x,n_y) = L(n_x, ops1_x) & L(ops1_x,ops1_y) = L(n_y, ops1_y)
            index = (L[n_x, n_y] == L[n_x, ops1_x]) & (
                L[ops1_x, ops1_y] == L[n_y, ops1_y])
            if len(np.where(index)[0]) == 0:
                cn_x = np.delete(cn_x, r)
                cn_y = np.delete(cn_y, r)

            else:
                ops2_x, ops2_y = ops1_x[index], ops1_y[index]
                # options of edges that didn't exist before
                index = [(newB[min(n_x, ops2_x[i])][max(n_x, ops2_x[i])] == 0)
                         & (newB[min(n_y, ops2_y[i])][max(n_y,
                                                          ops2_y[i])] == 0)
                         for i in range(len(ops2_x))]
                if (len(np.where(index)[0]) == 0):
                    cn_x = np.delete(cn_x, r)
                    cn_y = np.delete(cn_y, r)

                else:
                    ops3_x, ops3_y = ops2_x[index], ops2_y[index]

                    # choose randomly one edge from the final options
                    r1 = rs.randint(len(ops3_x))
                    nn_x, nn_y = ops3_x[r1], ops3_y[r1]

                    # Disconnect the existing edges
                    newB[n_x, n_y] = 0
                    newB[nn_x, nn_y] = 0
                    # Connect the new edges
                    newB[min(n_x, nn_x), max(n_x, nn_x)] = 1
                    newB[min(n_y, nn_y), max(n_y, nn_y)] = 1
                    # one successfull rewire!
                    nr += 1

                    # rewire with replacement
                    if replacement:
                        cn_x[r], cn_y[r] = min(n_x, nn_x), max(n_x, nn_x)
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x[index], cn_y[index] = min(
                            n_y, nn_y), max(n_y, nn_y)
                    # rewire without replacement
                    else:
                        cn_x = np.delete(cn_x, r)
                        cn_y = np.delete(cn_y, r)
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x = np.delete(cn_x, index)
                        cn_y = np.delete(cn_y, index)

    if nr < nswap:
        print(f"I didn't finish, out of rewirable edges: {len(cn_x)}")

    i, j = np.triu_indices(N, k=1)
    # Make the connectivity matrix symmetric
    newB[j, i] = newB[i, j]

    # check the number of edges is preserved
    if len(np.where(B != 0)[0]) != len(np.where(newB != 0)[0]):
        print(
            f"ERROR --- number of edges changed, \
            B:{len(np.where(B!=0)[0])}, newB:{len(np.where(newB!=0)[0])}")
    # check that the degree of the nodes it's the same
    for i in range(N):
        if np.sum(B[i]) != np.sum(newB[i]):
            print(
                f"ERROR --- node {i} changed k by: \
                {np.sum(B[i]) - np.sum(newB[i])}")

    newW = np.zeros((N, N))
    if weighted:
        # Reassign the weights
        mask = np.triu(B != 0, k=1)
        inids = D[mask]
        iniws = W[mask]
        inids_index = np.argsort(inids)
        # Weights from the shortest to largest edges
        iniws = iniws[inids_index]
        mask = np.triu(newB != 0, k=1)
        finds = D[mask]
        i, j = np.where(mask)
        # Sort the new edges from the shortest to the largest
        finds_index = np.argsort(finds)
        i_sort = i[finds_index]
        j_sort = j[finds_index]
        # Assign the initial sorted weights
        newW[i_sort, j_sort] = iniws
        # Make it symmetrical
        newW[j_sort, i_sort] = iniws

    return newB, newW, nr


def randmio_und(W, itr):
    """
    Optimized version of randmio_und.

    This function randomizes an undirected network, while preserving the
    degree distribution. The function does not preserve the strength
    distribution in weighted networks.

    This function is significantly faster if numba is enabled, because
    the main overhead is `np.random.randint`, see `here <https://stackoverflow.com/questions/58124646/why-in-python-is-random-randint-so-much-slower-than-random-random>`_

    Parameters
    ----------
    W : (N, N) array-like
        Undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.

    Returns
    -------
    W : (N, N) array-like
        Randomized network
    eff : int
        number of actual rewirings carried out
    """  # noqa: E501
    W = W.copy()
    n = len(W)
    i, j = np.where(np.triu(W > 0, 1))
    k = len(i)
    itr *= k

    # maximum number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))
    # actual number of successful rewirings
    eff = 0

    for _ in range(int(itr)):
        att = 0
        while att <= max_attempts:  # while not rewired
            while True:
                e1, e2 = np.random.randint(k), np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a, b = i[e1], j[e1]
                c, d = i[e2], j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            # flip edge c-d with 50% probability
            # to explore all potential rewirings
            if np.random.random() > .5:
                i[e2], j[e2] = d, c
                c, d = d, c

            # rewiring condition
            # not flipped
            # a--b    a  b
            #      TO  X
            # c--d    c  d
            # if flipped
            # a--b    a--b    a  b
            #      TO      TO  X
            # c--d    d--c    d  c
            if not (W[a, d] or W[c, b]):
                W[a, d] = W[a, b]
                W[a, b] = 0
                W[d, a] = W[b, a]
                W[b, a] = 0
                W[c, b] = W[c, d]
                W[c, d] = 0
                W[b, c] = W[d, c]
                W[d, c] = 0

                j[e1] = d
                j[e2] = b  # reassign edge indices
                eff += 1
                break
            att += 1

    return W, eff


if use_numba:
    randmio_und = njit(randmio_und)
