# -*- coding: utf-8 -*-
"""
Functions for generating group-level networks from individual measurements
"""

import numpy as np
from scipy.sparse import csgraph
from sklearn.utils.validation import (check_random_state, check_array,
                                      check_consistent_length)

from . import utils


def func_consensus(data, n_boot=1000, ci=95, seed=None):
    """
    Calculates thresholded group consensus functional connectivity graph

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
    data : (N, T, S) array_like
        Pre-processed functional time series, where `N` is the number of nodes,
        `T` is the number of volumes in the time series, and `S` is the number
        of subjects
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
        corrs = [np.corrcoef(data[..., sub]) for sub in range(data.shape[-1])]
        return np.mean(corrs, axis=0)

    collapsed_data = data.reshape((len(data), -1), order='F')
    consensus = np.corrcoef(collapsed_data)

    # only keep the upper triangle for the bootstraps to save on memory usage
    triu_inds = np.triu_indices_from(consensus, k=1)
    bootstrapped_corrmat = np.zeros((len(triu_inds[0]), n_boot))

    # generate `n_boot` bootstrap correlation matrices by sampling `t` time
    # points from the concatenated time series
    for boot in range(n_boot):
        inds = rs.randint(collapsed_data.shape[-1], size=data.shape[1])
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
    Estimates empirical cumulative distribution function of `data`

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


def struct_consensus(data, distance, hemiid):
    """
    Calculates distance-dependent group consensus structural connectivity graph

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

    Returns
    -------
    consensus : (N, N) numpy.ndarray
        Binary, group-level connectivity matrix

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
                         'data using array.reshape(-1, 1) and try again.')

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

    return consensus


def binarize_network(network, retain=10, keep_diag=False):
    """
    Keeps top `retain` % of connections in `network` and binarizes

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
    Keeps top `retain` % of connections in `network` and binarizes

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
