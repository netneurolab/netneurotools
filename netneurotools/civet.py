# -*- coding: utf-8 -*-
"""
Functions for working with CIVET data (ugh)
"""

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from .datasets import fetch_civet, fetch_fsaverage

_MNI305to152 = np.array([[0.9975, -0.0073, 0.0176, -0.0429],
                         [0.0146, 1.0009, -0.0024, 1.5496],
                         [-0.0130, -0.0093, 0.9971, 1.1840]])


def read_civet(fname):
    """
    Reads a CIVET-style .obj geometry file

    Parameters
    ----------
    fname : str or os.PathLike
        Filepath to .obj file

    Returns
    -------
    vertices : (N, 3)
    triangles : (T, 3)
    """

    k, polygons = 0, []
    with open(fname, 'r') as src:
        n_vert = int(src.readline().split()[6])
        vertices = np.zeros((n_vert, 3))
        for i, line in enumerate(src):
            if i < n_vert:
                vertices[i] = [float(i) for i in line.split()]
            elif i >= (2 * n_vert) + 5:
                if not line.strip():
                    k = 1
                elif k == 1:
                    polygons.extend([int(i) for i in line.split()])

    triangles = np.reshape(np.asarray(polygons), (-1, 3))

    return vertices, triangles


def _get_civet_to_fs_mapping(obj, fs):
    """
    Returns a mapping between `obj` and `fs` geometry files

    Parameters
    ----------
    obj : str or os.PathLike
        Path to CIVET geometry file
    fs : str or os.PathLike
        Path to FreeSurfer geometry file

    Returns
    -------
    idx : (N,) np.ndarray
        Mapping from CIVET to FreeSurfer space
    """

    vert_cv, _ = read_civet(obj)
    vert_fs, _ = nib.freesurfer.read_geometry(fs)

    vert_fs = np.c_[vert_fs, np.ones(len(vert_fs))] @ _MNI305to152.T
    _, idx = cKDTree(vert_cv).query(vert_fs, k=1, distance_upper_bound=10)

    return idx


def civet_to_freesurfer(brainmap, surface='mid', version='v1',
                        freesurfer='fsaverage6', mapping=None, data_dir=None):
    """
    Projects `brainmap` in CIVET space to `freesurfer` fsaverage space

    Uses a nearest-neighbor projection based on the geometry of the vertices

    Parameters
    ----------
    brainmap : array_like
        CIVET brainmap to be converted to freesurfer space
    surface : {'white', 'mid'}, optional
        Which CIVET surface to use for geometry of `brainmap`. Default: 'mid'
    version : {'v1', 'v2'}, optional
        Which CIVET version to use for geometry of `brainmap`. Default: 'v1'
    freesurfer : str, optional
        Which version of FreeSurfer space to project data to. Must be one of
        {'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6'}.
        Default: 'fsaverage6'
    mapping : array_like, optional
        If mapping has been pre-computed for `surface` --> `version` and is
        provided, this will be used instead of recomputing. Default: None
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None

    Returns
    -------
    data : np.ndarray
        Provided `brainmap` mapped to FreeSurfer
    """

    densities = (81924, 327684)
    n_vert = len(brainmap)
    if n_vert not in densities:
        raise ValueError('Unable to interpret `brainmap` space; provided '
                         'array must have length in {}. Received: {}'
                         .format(densities, n_vert))

    n_vert = n_vert // 2
    icbm = fetch_civet(density='41k' if n_vert == 40962 else '164k',
                       version=version, data_dir=data_dir, verbose=0)[surface]
    fsavg = fetch_fsaverage(version=freesurfer, data_dir=data_dir, verbose=0)
    fsavg = fsavg['pial' if surface == 'mid' else 'white']

    data = []
    for n, hemi in enumerate(('lh', 'rh')):
        sl = slice(n_vert * n, n_vert * (n + 1))
        if mapping is None:
            idx = _get_civet_to_fs_mapping(getattr(icbm, hemi),
                                           getattr(fsavg, hemi))
        else:
            idx = mapping[sl]

        data.append(brainmap[sl][idx])

    return np.hstack(data)
