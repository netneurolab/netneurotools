# -*- coding: utf-8 -*-
"""
Functions for clustering and working with cluster solutions
"""

import bct
import numpy as np
from scipy import optimize
from scipy.cluster import hierarchy
from sklearn.utils.validation import check_random_state


def _get_relabels(c1, c2):
    """
    Finds mapping of labels from `c1` to `c2`

    Parameters
    ----------
    c1, c2 : (N,) array_like
        Cluster labels for `N` subjects

    Returns
    -------
    src, tar : numpy.ndarray
        Source-target mapping of labels in `c1`
    """

    def _match(c1s, c2s):
        intersect = len(np.intersect1d(c1s, c2s))
        return (len(c1s) - intersect) + (len(c2s) - intersect)

    # get unique IDs of clusters in both solutions
    ids1, ids2 = np.unique(c1), np.unique(c2)

    idxs = np.arange(len(c1))
    assignments = np.ones((len(ids1), len(ids2)), dtype=int) * -1

    for n, i in enumerate(ids1):
        c1s = idxs[c1 == i]
        assignments[n] = [_match(c1s, idxs[c2 == f]) for f in ids2]

    idx1, idx2 = optimize.linear_sum_assignment(assignments)

    return ids1[idx1], ids2[idx2]


def match_cluster_labels(source, target):
    """
    Aligns cluster labels in `source` to those in `target`

    Uses :func:`scipy.optimize.linear_sum_assignment` to match solutions. If
    `source` has fewer clusters than `target` the returned assignments may be
    discontinuous (see Examples for more information).

    Parameters
    ----------
    source : (N,) array_like
        Cluster labels for `N` subjects, to be re-labelled
    target : (N,) array_like
        Cluster labels for `N` subjects, to which `source` is mapped

    Returns
    -------
    matched : (N,) array_like
        Re-labelled `source` with cluster assignments "matched" to `target`

    Examples
    --------
    >>> from netneurotools import cluster

    When cluster labels are perfectly matched but e.g., inverted the function
    will find a perfect mapping:

    >>> a = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    >>> b = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    >>> cluster.match_cluster_labels(a, b)
    array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    However, the mapping will work even when cluster assignments between the
    two solutions aren't perfectly matched. The function will simply choose a
    re-labelling that generates the "best" alignment between labels:

    >>> a = np.array([0, 0, 0, 2, 2, 2, 2, 1, 1, 1])
    >>> b = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    >>> cluster.match_cluster_labels(a, b)
    array([1, 1, 1, 0, 0, 0, 0, 2, 2, 2])

    If the source assignment has fewer clusters than the target the returned
    values may be discontinuous:

    >>> cluster.match_cluster_labels(b, a)
    array([0, 0, 0, 2, 2, 2, 2, 2, 2, 2])
    """

    # try and match the source to target
    src, tar = _get_relabels(source, target)

    # if there are a different number of clusters then handle elegantly.
    # elegantly here means we renumber the clusters so that they start at 1
    src_m = np.setdiff1d(np.unique(source), src)
    if len(src_m) > 0:
        tar_m = np.arange(tar.max() + 1, tar.max() + 1 + len(src_m))
        src, tar = np.append(src, src_m), np.append(tar, tar_m)

    # now re-label things based on the matched assignments
    sidx = src.argsort()
    src, tar = src[sidx], tar[sidx]
    matched = tar[np.searchsorted(src, source)]

    return matched


def match_assignments(assignments, target=None, seed=None):
    """
    Re-labels clusters in columns of `assignments` to best match `target`

    Uses :func:`~.cluster.match_cluster_labels` to align cluster assignments.

    Parameters
    ----------
    assignments : (N, M) array_like
        Array of `M` clustering assignments for `N` subjects
    target : (N,) array_like, optional
        Target clustering assignments to which all columns should be matched.
        If provided as an integer the relevant column in `assignments` will be
        selected. If not specified a (semi-)random column in `assignments` is
        chosen; because of the potential discontinuity introduced when matching
        an N-cluster solution to an N+1-cluster solution, the "random" target
        columns will be one `assignments` with the lowest cluster number. See
        Examples for more information. Default: None
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation; only used if `target` is not
        provided. Default: None

    Returns
    -------
    assignments : (N, M) numpy.ndarray
        Provided array with re-labeled cluster solutions to better match across
        `M` assignments

    Examples
    --------
    >>> from netneurotools import cluster

    First we can construct a matrix of `N` samples clustered `M` times (in this
    case, `M` is three) . Since cluster labels are generally arbitrary we can
    see that, while the same clusters were found each time, they were given
    different labels:

    >>> assignments = np.array([[0, 0, 1],
    ...                         [0, 0, 1],
    ...                         [0, 0, 1],
    ...                         [1, 2, 0],
    ...                         [1, 2, 0],
    ...                         [1, 2, 0],
    ...                         [2, 1, 2],
    ...                         [2, 1, 2]])

    We would like to match the assignments so they're all the same. Since one
    of the columns will be randomly picked as the "target" solution, we provide
    a `seed` to ensure reproducibility in the selection:

    >>> cluster.match_assignments(assignments, seed=1234)
    array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1],
           [0, 0, 0],
           [0, 0, 0],
           [0, 0, 0],
           [2, 2, 2],
           [2, 2, 2]])

    Alternatively, if `assignments` has clustering solutions with different
    numbers of clusters and no `target` is specified, the chosen `target` will
    be one of the columns with the smallest number of clusters:

    >>> assignments = np.array([[0, 0, 1],
    ...                         [0, 0, 1],
    ...                         [0, 0, 1],
    ...                         [1, 2, 0],
    ...                         [1, 2, 0],
    ...                         [1, 2, 0],
    ...                         [1, 1, 2],
    ...                         [1, 1, 2]])
    >>> cluster.match_assignments(assignments)
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0],
           [1, 1, 1],
           [1, 1, 1],
           [1, 1, 1],
           [1, 2, 2],
           [1, 2, 2]])
    """

    assignments = np.asarray(assignments).copy()

    # pick a random assignment with the lowest # of clusters as "target"
    if target is None:
        rs = check_random_state(seed)
        cm = assignments.max(axis=0)
        mask = cm == cm.min()
        target = assignments[:, mask][:, rs.choice(mask.sum())]
    # use the specified column of the matrix
    elif isinstance(target, int):
        target = assignments[:, target]
    # assume that target is an iterable we can use (just check the length)
    else:
        if len(target) != len(assignments):
            raise ValueError('Length of target clustering solution must be '
                             'identical to length of provided array.')

    # iterate through all assignments and try and match them to the target
    for n, source in enumerate(assignments.T):
        assignments[:, n] = match_cluster_labels(source, target)

    return assignments


def reorder_assignments(assignments, consensus=None, col_sort=True,
                        row_sort=True, return_index=True, seed=None):
    """
    Relabels and reorders rows / columns of `assignments` to "look better"

    Relabels cluster solutions in `assignments` so that distinct clustering
    solutions have similar cluster labels. Then, swaps columns of `assignments`
    so that similar clustering solutions are placed near each other. Finally,
    swaps rows of `assignments` so that subjects with similar clustering
    profiles are placed near each other.

    Uses hierarchical clustering to generate re-ordering of columns and rows

    Parameters
    ----------
    assignments : (N, M) array_like
        Array of `M` clustering assignments for `N` subjects
    consensus : (N,) array_like, optional
        "Final" clustering solution for `N` subjects. If provided, reordering
        of rows will be constrained by cluster assignment. Default: None
    {row,col}_sort : bool, optional
        If True, sort the {rows, columns}. Default: True
    return_index : bool, optional
        Whether to return the row and column indices used to re-order
        `assignments` in addition to the re-ordered matrix. Default: True
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    reordered : (N, M) numpy.ndarray
        Provided array with both rows and columns re-ordered
    index : tuple
        Indices used to reorder `assignments` to generate `reordered` output
    """

    def _reorder_rows(arr):
        """ Returns indices of rows in `arr` after hierarchical clustering
        """
        link = hierarchy.linkage(arr, method='average', metric='hamming')
        return hierarchy.dendrogram(link, no_plot=True)['leaves']

    # first, relabel the columns to try and match across assignments; this will
    # make our reordering procedure work a bit better!
    assignments = match_assignments(assignments, seed=seed)

    if col_sort:
        # get max cluster number for each partition
        max_cl = assignments.max(axis=0)

        # if different assignments have different numbers of detected clusters
        if len(np.unique(max_cl)) > 1:
            # first sort based on the number of clusters in each assignment
            col_idx = max_cl.argsort()
            assignments, max_cl = assignments[:, col_idx], max_cl[col_idx]

            # then, within assignments with the same number of clusters reorder
            # assignments (columns)
            reordered, splits = [], np.where(np.diff(max_cl) != 0)[0] + 1
            col_idx = np.split(col_idx, splits)
            for n, cl in enumerate(np.split(assignments, splits, axis=1)):
                idx = _reorder_rows(cl.T)
                col_idx[n] = col_idx[n][idx]
                reordered += [cl[:, idx]]
            col_idx = list(np.hstack(col_idx))
            assignments = np.column_stack(reordered)

        # otherwise all assignments have same number of detected clusters so
        # just sort them all
        else:
            col_idx = list(_reorder_rows(assignments.T))
            assignments = assignments[:, col_idx]

    if row_sort:
        # if a consensus was provided reorder rows based on cluster assignment
        if consensus is not None:
            # sort subjects by their cluster assignment in the consensus for
            # each cluster, then reorder subjects (rows)
            reordered, row_idx = [], []
            for cl in np.unique(consensus):
                cl, = np.where(consensus == cl)
                idx = list(cl[_reorder_rows(assignments[cl])])
                reordered += [assignments[idx]]
                row_idx += idx
            assignments = np.row_stack(reordered)

        # otherwise, just do a massive reordering of all the rows
        else:
            row_idx = list(_reorder_rows(assignments))
            assignments = assignments[row_idx]

    if return_index:
        return assignments, np.ix_(row_idx, col_idx)

    return assignments


def find_consensus(assignments, null_func=np.mean, return_agreement=False,
                   seed=None):
    """
    Finds consensus clustering labels from cluster solutions in `assignments`

    Parameters
    ----------
    assignments : (N, M) array_like
        Array of `M` clustering solutions for `N` samples (e.g., subjects,
        nodes, etc). Values of array should be integer-based cluster assignment
        labels
    null_func : callable, optional
        Function used to generate null model when performing consensus-based
        clustering. Must accept a 2D array as input and return a single value.
        Default: :func:`numpy.mean`
    return_agreement : bool, optional
        Whether to return the thresholded N x N agreement matrix used in
        generating the final consensus clustering solution. Default: False
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Used when permuting cluster
        assignments during generation of null model. Default: None

    Returns
    -------
    consensus : (N,) numpy.ndarray
        Consensus cluster labels

    References
    ----------
    Bassett, D. S., Porter, M. A., Wymbs, N. F., Grafton, S. T., Carlson,
    J. M., & Mucha, P. J. (2013). Robust detection of dynamic community
    structure in networks. Chaos: An Interdisciplinary Journal of Nonlinear
    Science, 23(1), 013142.
    """

    rs = check_random_state(seed)
    samp, comm = assignments.shape

    # create agreement matrix from input community assignments and convert to
    # probability matrix by dividing by `comm`
    agreement = bct.clustering.agreement(assignments, buffsz=samp) / comm

    # generate null agreement matrix and use to create threshold
    null_assign = np.column_stack([rs.permutation(i) for i in assignments.T])
    null_agree = bct.clustering.agreement(null_assign, buffsz=samp) / comm
    threshold = null_func(null_agree)

    # run consensus clustering on agreement matrix after thresholding
    consensus = bct.clustering.consensus_und(agreement, threshold, 10)

    if return_agreement:
        return consensus.astype(int), agreement * (agreement > threshold)

    return consensus.astype(int)
