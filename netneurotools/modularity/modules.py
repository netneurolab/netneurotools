"""Functions for working with network modules."""

import bct
import numpy as np
from sklearn.utils.validation import check_random_state
from scipy import optimize
from scipy.cluster import hierarchy

from .. import has_numba
if has_numba:
    from numba import njit


def _get_relabels(c1, c2):
    """
    Find mapping of labels from `c1` to `c2`.

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
    Align cluster labels in `source` to those in `target`.

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
    >>> from netneurotools import modularity

    When cluster labels are perfectly matched but e.g., inverted the function
    will find a perfect mapping:

    >>> a = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    >>> b = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    >>> modularity.match_cluster_labels(a, b)
    array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    However, the mapping will work even when cluster assignments between the
    two solutions aren't perfectly matched. The function will simply choose a
    re-labelling that generates the "best" alignment between labels:

    >>> a = np.array([0, 0, 0, 2, 2, 2, 2, 1, 1, 1])
    >>> b = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    >>> modularity.match_cluster_labels(a, b)
    array([1, 1, 1, 0, 0, 0, 0, 2, 2, 2])

    If the source assignment has fewer clusters than the target the returned
    values may be discontinuous:

    >>> modularity.match_cluster_labels(b, a)
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
    Re-label clusters in columns of `assignments` to best match `target`.

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
    >>> from netneurotools import modularity

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

    >>> modularity.match_assignments(assignments, seed=1234)
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
    >>> modularity.match_assignments(assignments)
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
    Relabel and reorders rows / columns of `assignments` to "look better".

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
        """Return indices of rows in `arr` after hierarchical clustering."""
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
            assignments = np.vstack(reordered)

        # otherwise, just do a massive reordering of all the rows
        else:
            row_idx = list(_reorder_rows(assignments))
            assignments = assignments[row_idx]

    if return_index:
        return assignments, np.ix_(row_idx, col_idx)

    return assignments


def agreement_matrix(assignments):
    """
    Compute the agreement matrix from cluster solutions in `assignments`.

    Parameters
    ----------
    assignments : (N, M) array_like
        Array of `M` clustering solutions for `N` samples (e.g., subjects,
        nodes, etc). Values of array should be integer-based cluster assignment
        labels

    Returns
    -------
    agreement: (N, N) array_like
        Agreement matrix where each entry indicates the number of times a pair
        of samples was assigned to the same cluster.
    """
    assignments = np.asarray(assignments)
    N, M = assignments.shape
    agreement = np.zeros((N, N), dtype=np.int32)

    for k in range(M):
        labels = assignments[:, k]
        for lbl in np.unique(labels):
            nodes = np.where(labels == lbl)[0]
            agreement[np.ix_(nodes, nodes)] += 1

    np.fill_diagonal(agreement, 0)
    return agreement


def consensus_clustering(agreement, threshold, n_it=10):
    """
    Perform consensus clustering on an agreement matrix.

    This algorithm, first introduced in [1]_, iteratively thresholds the
    agreement matrix and applies community detection to identify stable
    clustering partitions. The procedure terminates when a single unique
    partition is obtained or when no supra-threshold agreement remains.

    Parameters
    ----------
    agreement : (N, N) array_like
        Agreement matrix indicating the frequency with which pairs of
        samples were assigned to the same cluster
    threshold : float
        Threshold value used to remove weak agreement entries
    n_it : int, optional
        Number of community detection iterations per consensus step.
        Default: 10

    Returns
    -------
    consensus : (N,) numpy.ndarray
        Consensus cluster labels

    Notes
    -----
    This algorithm is adapted from `consensus_und` of the Brain Connectivity
    Toolbox [2]_.

    References
    ----------
    .. [1] Lancichinetti, A., & Fortunato, S. (2012). Consensus clustering in
       complex networks. Scientific reports, 2(1), 336.

    .. [2] Rubinov, M., & Sporns, O. (2010). Complex network measures of brain
       connectivity: uses and interpretations. Neuroimage, 52(3), 1059-1069.

    """
    n = len(agreement)
    while True:

        thresholded_agreement = np.where(agreement >= threshold, agreement, 0)
        np.fill_diagonal(thresholded_agreement, 0)

        if np.all(thresholded_agreement == 0):
            ci_unique = np.arange(n)
            break
        else:
            ci = np.zeros((n, n_it))
            for i in range(n_it):
                ci[:, i], _ = bct.community_louvain(thresholded_agreement,
                                                    B='modularity')
            ci_unique = _unique_partitions(ci)
            if ci_unique.shape[1] > 1:
                agreement = agreement_matrix(ci) / n_it
            else:
                break

    return np.squeeze(ci_unique + 1)


def find_consensus(assignments, null_func=np.mean, return_agreement=False,
                   seed=None):
    """
    Find consensus clustering labels from cluster solutions in `assignments`.

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
    agreement = agreement_matrix(assignments) / comm

    # generate null agreement matrix and use to create threshold
    null_assign = np.column_stack([rs.permutation(i) for i in assignments.T])
    null_agree = agreement_matrix(null_assign) / comm
    threshold = null_func(null_agree)

    # run consensus clustering on agreement matrix after thresholding
    consensus = consensus_clustering(agreement, threshold, n_it=10)

    if return_agreement:
        return consensus.astype(int), agreement * (agreement > threshold)

    return consensus.astype(int)


def consensus_modularity(adjacency, gamma=1, B='modularity',
                         repeats=250, null_func=np.mean, seed=None):
    """
    Find community assignments from `adjacency` through consensus.

    Performs `repeats` iterations of community detection on `adjacency` and
    then uses consensus clustering on the resulting community assignments.

    Parameters
    ----------
    adjacency : (N, N) array_like
        Adjacency matrix (weighted/non-weighted) on which to perform consensus
        community detection.
    gamma : float, optional
        Resolution parameter for modularity maximization. Default: 1
    B : str or (N, N) array_like, optional
        Null model to use for consensus clustering. If `str`, must be one of
        ['modularity', 'potts', 'negative_sym', 'negative_asym']. Default:
        'modularity'
    repeats : int, optional
        Number of times to repeat Louvain algorithm clustering. Default: 250
    null_func : callable, optional
        Function used to generate null model when performing consensus-based
        clustering. Must accept a 2D array as input and return a single value.
        Default: `np.mean`
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    consensus : (N,) np.ndarray
        Consensus-derived community assignments
    Q_all : array_like
        Optimized modularity over all `repeats` community assignments
    zrand_all : array_like
        z-Rand score over all pairs of `repeats` community assignment vectors

    References
    ----------
    Bassett, D. S., Porter, M. A., Wymbs, N. F., Grafton, S. T., Carlson,
    J. M., & Mucha, P. J. (2013). Robust detection of dynamic community
    structure in networks. Chaos: An Interdisciplinary Journal of Nonlinear
    Science, 23(1), 013142.
    """
    # generate community partitions `repeat` times
    comms, Q_all = zip(*[bct.community_louvain(adjacency, gamma=gamma, B=B)
                         for i in range(repeats)])
    comms = np.column_stack(comms)

    # find consensus cluster assignments across all partitoning solutions
    consensus = find_consensus(comms, null_func=null_func, seed=seed)

    # get z-rand statistics for partition similarity (n.b. can take a while)
    zrand_all = _zrand_partitions(comms)

    return consensus, np.array(Q_all), zrand_all


def zrand(X, Y):
    """
    Calculate the z-Rand index of two community assignments.

    Communities are relabeled to consecutive integers for the comparison.

    Parameters
    ----------
    X, Y : (n, 1) array_like
        Community assignment vectors to compare

    Returns
    -------
    z_rand : float
        Z-rand index

    References
    ----------
    Amanda L. Traud, Eric D. Kelsic, Peter J. Mucha, and Mason A. Porter.
    (2011). Comparing Community Structure to Characteristics in Online
    Collegiate Social Networks. SIAM Review, 53, 526-543.
    """
    X = np.squeeze(X)
    Y = np.squeeze(Y)

    if X.ndim > 1 or Y.ndim > 1:
        raise ValueError('X and Y must have only one-dimension each. '
                         'Please check inputs.')

    if len(X) != len(Y):
        raise ValueError('X and Y must have the same length.')

    X = np.unique(X, return_inverse=True)[1]
    Y = np.unique(Y, return_inverse=True)[1]

    # Calculate the contigency table + calculate row/column marginals
    kx = X.max() + 1
    ky = Y.max() + 1
    nij = np.bincount(X * ky + Y).reshape(kx, ky)
    ni = nij.sum(axis=1)
    nj = nij.sum(axis=0)

    n = len(X)
    M = n * (n - 1) // 2
    M1 = np.sum(ni * (ni - 1)) / 2
    M2 = np.sum(nj * (nj - 1)) / 2
    wab = np.sum(nij * (nij - 1)) / 2

    mod = n * (n**2 - 3 * n - 2)

    C1 = mod - 8 * (n + 1) * M1 + 4 * np.sum(ni**3)
    C2 = mod - 8 * (n + 1) * M2 + 4 * np.sum(nj**3)

    a = M / 16
    b = ((4 * M1 - 2 * M)**2) * ((4 * M2 - 2 * M)**2) / (256 * (M**2))
    c = C1 * C2 / (16 * n * (n - 1) * (n - 2))
    d = ((((4 * M1 - 2 * M)**2) - (4 * C1) - (4 * M))
         * (((4 * M2 - 2 * M)**2) - (4 * C2) - (4 * M))
         / (64 * n * (n - 1) * (n - 2) * (n - 3)))

    sigw2 = a - b + c + d
    # catch any negatives
    if sigw2 < 0:
        return 0
    z_rand = (wab - ((M1 * M2) / M)) / np.sqrt(sigw2)

    return z_rand


def _zrand_partitions(communities):
    """
    Calculate z-Rand for all pairs of assignments in `communities`.

    Iterates through every pair of community assignment vectors in
    `communities` and calculates the z-Rand score to assess their similarity.

    Parameters
    ----------
    communities : (S, R) array_like
        Community assignments for `S` samples over `R` partitions

    Returns
    -------
    all_zrand : array_like
        z-Rand score over all pairs of `R` partitions of community assignments
    """
    n_partitions = communities.shape[-1]
    all_zrand = np.zeros(int(n_partitions * (n_partitions - 1) / 2))

    for c1 in range(n_partitions):
        for c2 in range(c1 + 1, n_partitions):
            idx = int((c1 * n_partitions) + c2 - ((c1 + 1) * (c1 + 2) // 2))
            all_zrand[idx] = zrand(communities[:, c1], communities[:, c2])

    return all_zrand


def _unique_partitions(communities):
    """
    Identify unique clustering partitions from a set of solutions.

    Relabels each partition in `communities` to use consecutive integer labels
    starting from zero, then removes duplicate partitions across columns.

    Parameters
    ----------
    communities : (N, M) array_like
        Array of `M` clustering solutions for `N` samples

    Returns
    -------
    unique : (N, K) numpy.ndarray
        Array containing the `K` unique clustering partitions identified
        from `communities`
    """
    # Relabel partitions to consecutive integers
    n, m = communities.shape
    for i in range(m):
        communities[:, i] = np.unique(communities[:, i], return_inverse=True)[1]

    # Remove duplicate partitions
    seen = set()
    unique_cols = []
    for i in range(m):
        key = tuple(communities[:, i])
        if key not in seen:
            seen.add(key)
            unique_cols.append(communities[:, i])

    return np.column_stack(unique_cols)


if has_numba:
    _zrand_partitions = njit(_zrand_partitions, parallel=True)


def get_modularity(adjacency, comm, gamma=1):
    """
    Calculate modularity contribution for each community in `comm`.

    Parameters
    ----------
    adjacency : (N, N) array_like
        Adjacency (e.g., correlation) matrix
    comm : (N,) array_like
        Community assignment vector splitting `N` subjects into `G` groups
    gamma : float, optional
        Resolution parameter used in original modularity maximization.
        Default: 1

    Returns
    -------
    comm_q : (G,) ndarray
        Relative modularity for each community

    See Also
    --------
    netneurotools.modularity.get_modularity_z
    netneurotools.modularity.get_modularity_sig
    """
    adjacency, comm = np.asarray(adjacency), np.asarray(comm)
    s = adjacency.sum()
    B = adjacency - (gamma * np.outer(adjacency.sum(axis=1),
                                      adjacency.sum(axis=0)) / s)

    # find modularity contribution of each community
    communities = np.unique(comm)
    comm_q = np.empty(shape=communities.size)
    for n, ci in enumerate(communities):
        inds = comm == ci
        comm_q[n] = B[np.ix_(inds, inds)].sum() / s

    return comm_q


def get_modularity_z(adjacency, comm, gamma=1, n_perm=10000, seed=None):
    """
    Calculate average z-score of community assignments by permutation.

    Parameters
    ----------
    adjacency : (N, N) array_like
        Adjacency (correlation) matrix
    comm : (N,) array_like
        Community assignment vector splitting `N` subjects into `G` groups
    gamma : float, optional
        Resolution parameter used in original modularity maximization.
        Default: 1
    n_perm : int, optional
        Number of permutations. Default: 10000
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    q_z : float
        Average Z-score of modularity of communities

    See Also
    --------
    netneurotools.modularity.get_modularity
    netneurotools.modularity.get_modularity_sig
    """
    rs = check_random_state(seed)

    real_qs = get_modularity(adjacency, comm, gamma)
    simu_qs = np.empty(shape=(np.unique(comm).size, n_perm))
    for perm in range(n_perm):
        simu_qs[:, perm] = get_modularity(adjacency,
                                          rs.permutation(comm),
                                          gamma)

    # avoid instances where dist.std(1) == 0
    std = simu_qs.std(axis=1)
    if np.any(std == 0):
        return np.mean(real_qs - simu_qs.mean(axis=1))
    else:
        return np.mean((real_qs - simu_qs.mean(axis=1)) / std)


def get_modularity_sig(adjacency, comm, gamma=1, n_perm=10000, alpha=0.01,
                       seed=None):
    """
    Calculate significance of community assignments in `comm` by permutation.

    Parameters
    ----------
    adjacency : (N, N) array_like
        Adjacency (correlation) matrix
    comm : (N,) array_like
        Community assignment vector
    gamma : float
        Resolution parameter used in original modularity maximization
    n_perm : int, optional
        Number of permutations to test against. Default: 10000
    alpha : (0,1) float, optional
        Alpha level to assess significance. Default: 0.01
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    ndarray
        Significance of each community in `comm` (boolean)

    See Also
    --------
    netneurotools.modularity.get_modularity_z
    netneurotools.modularity.get_modularity_sig
    """
    rs = check_random_state(seed)

    real_qs = get_modularity(adjacency, comm, gamma)
    simu_qs = np.empty(shape=(np.unique(comm).size, n_perm))
    for perm in range(n_perm):
        simu_qs[:, perm] = get_modularity(adjacency,
                                          rs.permutation(comm),
                                          gamma)

    q_sig = real_qs > np.percentile(simu_qs, 100 * (1 - alpha), axis=1)

    return q_sig
