"""
Functions for constructing graphs from surface meshes
"""

import numpy as np
from scipy import sparse


def _get_edges(faces):
    """
    Gets set of edges from `faces`

    Parameters
    ----------
    faces : (F, 3) array_like
        Set of indices creating triangular faces of a mesh

    Returns
    -------
    edges : (F*3, 2) array_like
        All edges in `faces`
    """

    faces = np.asarray(faces)
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2)), axis=1)

    return edges


def get_direct_edges(vertices, faces):
    """
    Gets (unique) direct edges and weights in mesh describes by inputs.

    Parameters
    ----------
    vertices : (N, 3) array_like
        Coordinates of `vertices` comprising mesh with `faces`
    faces : (F, 3) array_like
        Indices of `vertices` that compose triangular faces of mesh

    Returns
    -------
    edges : (E, 2) array_like
        Indices of `vertices` comprising direct edges (without duplicates)
    weights : (E, 1) array_like
        Distances between `edges`

    """
    edges = np.unique(_get_edges(faces), axis=0)
    weights = np.linalg.norm(np.diff(vertices[edges], axis=1), axis=-1)
    return edges, weights.squeeze()


def get_indirect_edges(vertices, faces):
    """
    Gets indirect edges and weights in mesh described by inputs

    Indirect edges are between two vertices that participate in faces sharing
    an edge

    Parameters
    ----------
    vertices : (N, 3) array_like
        Coordinates of `vertices` comprising mesh with `faces`
    faces : (F, 3) array_like
        Indices of `vertices` that compose triangular faces of mesh

    Returns
    -------
    edges : (E, 2) array_like
        Indices of `vertices` comprising indirect edges (without duplicates)
    weights : (E, 1) array_like
        Distances between `edges` on surface

    References
    ----------
    https://github.com/mikedh/trimesh (MIT licensed)

    """
    # first generate the list of edges for the provided faces and the
    # index for which face the edge is from (which is just the index of the
    # face repeated thrice, since each face generates three direct edges)
    edges = _get_edges(faces)
    edges_face = np.repeat(np.arange(len(faces)), 3)

    # every edge appears twice in a watertight surface, so we'll first get the
    # indices for each duplicate edge in `edges` (this should, assuming all
    # goes well, have rows equal to len(edges) // 2)
    order = np.lexsort(edges.T[::-1])
    edges_sorted = edges[order]
    dupe = np.any(edges_sorted[1:] != edges_sorted[:-1], axis=1)
    dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
    start_ok = np.diff(np.concatenate((dupe_idx, [len(edges_sorted)]))) == 2
    groups = np.tile(dupe_idx[start_ok].reshape(-1, 1), 2)
    edge_groups = order[groups + np.arange(2)]

    # now, get the indices of the faces that participate in these duplicate
    # edges, as well as the edges themselves
    adjacency = edges_face[edge_groups]
    nondegenerate = adjacency[:, 0] != adjacency[:, 1]
    adjacency = np.sort(adjacency[nondegenerate], axis=1)
    adjacency_edges = edges[edge_groups[:, 0][nondegenerate]]

    # the non-shared vertex index is the same shape as adjacency, holding
    # vertex indices vs face indices
    indirect_edges = np.zeros(adjacency.shape, dtype=np.int32) - 1

    # loop through the two columns of adjacency
    for i, fid in enumerate(adjacency.T):
        # faces from the current column of adjacency
        face = faces[fid]
        # get index of vertex not included in shared edge
        unshared = np.logical_not(np.logical_or(
            face == adjacency_edges[:, 0].reshape(-1, 1),
            face == adjacency_edges[:, 1].reshape(-1, 1)))
        # each row should have one "uncontained" vertex; ignore degenerates
        row_ok = unshared.sum(axis=1) == 1
        unshared[~row_ok, :] = False
        indirect_edges[row_ok, i] = face[unshared]

    # get vertex coordinates of triangles pairs with shared edges, ordered
    # such that the non-shared vertex is always _last_ among the trio
    shared = np.sort(face[np.logical_not(unshared)].reshape(-1, 1, 2), axis=-1)
    shared = np.repeat(shared, 2, axis=1)
    triangles = np.concatenate((shared, indirect_edges[..., None]), axis=-1)
    # `A.shape`: (3, N, 2) corresponding to (xyz coords, edges, triangle pairs)
    A, B, V = vertices[triangles].transpose(2, 3, 0, 1)

    # calculate the xyz coordinates of the foot of each triangle, where the
    # base is the shared edge
    # that is, we're trying to calculate F in the equation `VF = VB - (w * BA)`
    # where `VF`, `VB`, and `BA` are vectors, and `w = (AB * VB) / (AB ** 2)`
    w = (np.sum((A - B) * (V - B), axis=0, keepdims=True)
         / np.sum((A - B) ** 2, axis=0, keepdims=True))
    feet = B - (w * (B - A))
    # calculate coordinates of midpoint b/w the feet of each pair of triangles
    midpoints = (np.sum(feet.transpose(1, 2, 0), axis=1) / 2)[:, None]
    # calculate Euclidean distance between non-shared vertices and midpoints
    # and add distances together for each pair of triangles
    norms = np.linalg.norm(vertices[indirect_edges] - midpoints, axis=-1)
    weights = np.sum(norms, axis=-1)

    # NOTE: weights won't be perfectly accurate for a small subset of triangle
    # pairs where either triangle has angle >90 along the shared edge. in these
    # the midpoint lies _outside_ the shared edge, so neighboring triangles
    # would need to be taken into account. that said, this occurs in only a
    # minority of cases and the difference tends to be in the ~0.001 mm range
    return indirect_edges, weights


def make_surf_graph(vertices, faces, mask=None):
    """
    Constructs adjacency graph from `surf`.

    Parameters
    ----------
    vertices : (N, 3) array_like
        Coordinates of `vertices` comprising mesh with `faces`
    faces : (F, 3) array_like
        Indices of `vertices` that compose triangular faces of mesh
    mask : (N,) array_like, optional (default None)
        Boolean mask indicating which vertices should be removed from generated
        graph. If not supplied, all vertices are used.

    Returns
    -------
    graph : scipy.sparse.csr_matrix
        Sparse matrix representing graph of `vertices` and `faces`

    Raises
    ------
    ValueError : inconsistent number of vertices in `mask` and `vertices`
    """

    if mask is not None and len(mask) != len(vertices):
        raise ValueError('Supplied `mask` array has different number of '
                         'vertices than supplied `vertices`.')

    # get all (direct + indirect) edges from surface
    direct_edges, direct_weights = get_direct_edges(vertices, faces)
    indirect_edges, indirect_weights = get_indirect_edges(vertices, faces)
    edges = np.row_stack((direct_edges, indirect_edges))
    weights = np.hstack((direct_weights, indirect_weights))

    # remove edges that include a vertex in `mask`
    if mask is not None:
        idx, = np.where(mask)
        mask = ~np.any(np.isin(edges, idx), axis=1)
        edges, weights = edges[mask], weights[mask]

    # construct our graph on which to calculate shortest paths
    return sparse.csr_matrix((np.squeeze(weights), (edges[:, 0], edges[:, 1])),
                             shape=(len(vertices), len(vertices)))
