"""Functions for supporting network constuction."""

import numpy as np
from scipy.sparse import csgraph

def get_triu(data, k=1):
    """
    Return vectorized version of upper triangle from `data`.

    Parameters
    ----------
    data : (N, N) array_like
        Input data
    k : int, optional
        Which diagonal to select from (where primary diagonal is 0). Default: 1

    Returns
    -------
    triu : (N * N-1 / 2) numpy.ndarray
        Upper triangle of `data`

    Examples
    --------
    >>> from netneurotools import utils

    >>> X = np.array([[1, 0.5, 0.25], [0.5, 1, 0.33], [0.25, 0.33, 1]])
    >>> tri = utils.get_triu(X)
    >>> tri
    array([0.5 , 0.25, 0.33])
    """
    return data[np.triu_indices(len(data), k=k)].copy()


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
    triu = get_triu(network)
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
    inds = get_triu(graph).argsort()[:remain]
    inds = tuple(e[inds] for e in np.triu_indices_from(graph, k=1))

    # add edges to MST, symmetrize, and convert to binary matrix
    mst[inds] = graph[inds]
    mst = np.array((mst + mst.T) != 0, dtype=int)

    return mst