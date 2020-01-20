# -*- coding: utf-8 -*-
"""
Functions for working with FreeSurfer data and parcellations
"""

import os
import os.path as op

from nibabel.freesurfer import read_annot, read_geometry
import numpy as np
from scipy.ndimage.measurements import _stats, labeled_comprehension
from scipy.spatial.distance import cdist

from .datasets import fetch_fsaverage
from .stats import gen_spinsamples
from .utils import check_fs_subjid, run


def apply_prob_atlas(subject_id, gcs, hemi, *, orig='white', annot=None,
                     ctab=None, subjects_dir=None, use_cache=True,
                     quiet=False):
    """
    Creates an annotation file for `subject_id` by applying atlas in `gcs`

    Runs subprocess calling FreeSurfer's "mris_ca_label" function; as such,
    FreeSurfer must be installed and accesible on the local system path.

    Parameters
    ----------
    subject_id : str
        FreeSurfer subject ID
    gcs : str
        Filepath to .gcs file containing classifier array
    hemi : {'lh', 'rh'}
        Hemisphere corresponding to `gcs` file
    orig : str, optional
        Original surface to which to apply classifer. Default: 'white'
    annot : str, optional
        Path to output annotation file to generate. If set to None, the name is
        created from the provided `hemi` and `gcs`. If provided as a
        relative path, it is assumed to stem from `subjects_dir`/`subject_id`.
        Default: None
    ctab : str, optional
        Path to colortable corresponding to `gcs`. Default: None
    subjects_dir : str, optional
        Path to FreeSurfer subject directory. If not set, will inherit from
        the environmental variable $SUBJECTS_DIR. Default: None
    use_cache : bool, optional
        Whether to check for existence of `annot` in directory specified by
        `{subjects_dir}/{subject_id}/label' and use that, if it exists. If
        False, will create a new annot file. Default: True
    quiet : bool, optional
        Whether to restrict status messages. Default: False

    Returns
    -------
    annot : str
        Path to generated annotation file
    """

    cmd = 'mris_ca_label {opts}{subject_id} {hemi} {hemi}.sphere.reg ' \
          '{gcs} {annot}'

    if hemi not in ['rh', 'lh']:
        raise ValueError('Provided hemisphere designation `hemi` must be one '
                         'of \'rh\' or \'lh\'. Provided: {}'.format(hemi))
    if not op.isfile(gcs):
        raise ValueError('Cannot find specified `gcs` file {}.'.format(gcs))

    subject_id, subjects_dir = check_fs_subjid(subject_id, subjects_dir)

    # add all the options together, as specified
    opts = ''
    if ctab is not None and op.isfile(ctab):
        opts += '-t {} '.format(ctab)
    if orig is not None:
        opts += '-orig {} '.format(orig)
    if subjects_dir is not None:
        opts += '-sdir {} '.format(subjects_dir)
    else:
        subjects_dir = os.environ['SUBJECTS_DIR']

    # generate output filename
    if annot is None:
        base = '{}.{}.annot'.format(hemi, gcs[:-4])
        annot = op.join(subjects_dir, subject_id, 'label', base)
    else:
        # if not a full path, assume relative from subjects_dir/subject_id
        if not annot.startswith(op.abspath(os.sep)):
            annot = op.join(subjects_dir, subject_id, annot)

    # if annotation file doesn't exist or we explicitly want to make a new one
    if not op.isfile(annot) or not use_cache:
        run(cmd.format(opts=opts, subject_id=subject_id, hemi=hemi,
                       gcs=gcs, annot=annot),
            quiet=quiet)

    return annot


def _decode_list(vals):
    """ List decoder
    """

    return [l.decode() if hasattr(l, 'decode') else l for l in vals]


def find_parcel_centroids(*, lhannot, rhannot, version='fsaverage',
                          surf='sphere', drop=None):
    """
    Returns vertex coords corresponding to centroids of parcels in annotations

    Note that using any other `surf` besides the default of 'sphere' may result
    in centroids that are not directly within the parcels themselves due to
    sulcal folding patterns.

    Parameters
    ----------
    {lh,rh}annot : str
        Path to .annot file containing labels of parcels on the {left,right}
        hemisphere. These must be specified as keyword arguments to avoid
        accidental order switching.
    version : str, optional
        Specifies which version of `fsaverage` provided annotation files
        correspond to. Must be one of {'fsaverage', 'fsaverage3', 'fsaverage4',
        'fsaverage5', 'fsaverage6'}. Default: 'fsaverage'
    surf : str, optional
        Specifies which surface projection of fsaverage to use for finding
        parcel centroids. Default: 'sphere'
    drop : list, optional
        Specifies regions in {lh,rh}annot for which the parcel centroid should
        not be calculated. If not specified, centroids for 'unknown' and
        'corpuscallosum' are not calculated. Default: None

    Returns
    -------
    centroids : (N, 3) numpy.ndarray
        xyz coordinates of vertices closest to the centroid of each parcel
        defined in `lhannot` and `rhannot`
    hemiid : (N,) numpy.ndarray
        Array denoting hemisphere designation of coordinates in `centroids`,
        where `hemiid=0` denotes the right and `hemiid=1` the left hemisphere
    """

    if drop is None:
        drop = [
            'unknown', 'corpuscallosum',  # default FreeSurfer
            'Background+FreeSurfer_Defined_Medial_Wall'  # common alternative
        ]
    drop = _decode_list(drop)

    surfaces = fetch_fsaverage(version)[surf]

    centroids, hemiid = [], []
    for n, (annot, surf) in enumerate(zip([lhannot, rhannot], surfaces)):
        vertices, faces = read_geometry(surf)
        labels, ctab, names = read_annot(annot)
        names = _decode_list(names)

        for lab in np.unique(labels):
            if names[lab] in drop:
                continue
            coords = np.atleast_2d(vertices[labels == lab].mean(axis=0))
            roi = vertices[np.argmin(cdist(vertices, coords), axis=0)[0]]
            centroids.append(roi)
            hemiid.append(n)

    return np.row_stack(centroids), np.asarray(hemiid)


def parcels_to_vertices(data, *, lhannot, rhannot, drop=None):
    """
    Projects parcellated `data` to vertices defined in annotation files

    Assigns np.nan to all ROIs in `drop`

    Parameters
    ----------
    data : (N,) numpy.ndarray
        Parcellated data to be projected to vertices. Parcels should be ordered
        by [left, right] hemisphere; ordering within hemisphere should
        correspond to the provided annotation files.
    {lh,rh}annot : str
        Path to .annot file containing labels of parcels on the {left,right}
        hemisphere. These must be specified as keyword arguments to avoid
        accidental order switching.
    drop : list, optional
        Specifies regions in {lh,rh}annot that are not present in `data`. NaNs
        will be inserted in place of the these regions in the returned data. If
        not specified, 'unknown' and 'corpuscallosum' are assumed to not be
        present. Default: None

    Reurns
    ------
    projected : numpy.ndarray
        Vertex-level data
    """

    if drop is None:
        drop = [
            'unknown', 'corpuscallosum',  # default FreeSurfer
            'Background+FreeSurfer_Defined_Medial_Wall'  # common alternative
        ]
    drop = _decode_list(drop)

    data = np.vstack(data)

    # check this so we're not unduly surprised by anything...
    n_vert = expected = 0
    for a in [lhannot, rhannot]:
        vn, _, names = read_annot(a)
        n_vert += len(vn)
        names = _decode_list(names)
        expected += len(names) - len(set(drop) & set(names))
    if expected != len(data):
        raise ValueError('Number of parcels in provided annotation files '
                         'differs from size of parcellated data array.\n'
                         '    EXPECTED: {} parcels\n'
                         '    RECEIVED: {} parcels'
                         .format(expected, len(data)))

    projected = np.zeros((n_vert, data.shape[-1]), dtype=data.dtype)
    start = end = n_vert = 0
    for annot in [lhannot, rhannot]:
        # read files and update end index for `data`
        labels, ctab, names = read_annot(annot)
        names = _decode_list(names)
        todrop = set(names) & set(drop)
        end += len(names) - len(todrop)  # unknown and corpuscallosum

        # get indices of unknown and corpuscallosum and insert NaN values
        inds = sorted([names.index(f) for f in todrop])
        inds = [f - n for n, f in enumerate(inds)]
        currdata = np.insert(data[start:end], inds, np.nan, axis=0)

        # project to vertices and store
        projected[n_vert:n_vert + len(labels), :] = currdata[labels]
        start = end
        n_vert += len(labels)

    return np.squeeze(projected)


def vertices_to_parcels(data, *, lhannot, rhannot, drop=None):
    """
    Reduces vertex-level `data` to parcels defined in annotation files

    Takes average of vertices within each parcel, excluding np.nan values
    (i.e., np.nanmean). Assigns np.nan to parcels for which all vertices are
    np.nan.

    Parameters
    ----------
    data : (N,) numpy.ndarray
        Vertex-level data to be reduced to parcels
    {lh,rh}annot : str
        Path to .annot file containing labels to parcels on the {left,right}
        hemisphere
    drop : list, optional
        Specifies regions in {lh,rh}annot that should be removed from the
        parcellated version of `data`. If not specified, 'unknown' and
        'corpuscallosum' will be removed. Default: None

    Reurns
    ------
    reduced : numpy.ndarray
        Parcellated `data`, without regions specified in `drop`
    """

    if drop is None:
        drop = [
            'unknown', 'corpuscallosum',  # default FreeSurfer
            'Background+FreeSurfer_Defined_Medial_Wall'  # common alternative
        ]
    drop = _decode_list(drop)

    data = np.vstack(data)

    n_parc = expected = 0
    for a in [lhannot, rhannot]:
        vn, _, names = read_annot(a)
        expected += len(vn)
        names = _decode_list(names)
        n_parc += len(names) - len(set(drop) & set(names))
    if expected != len(data):
        raise ValueError('Number of vertices in provided annotation files '
                         'differs from size of vertex-level data array.\n'
                         '    EXPECTED: {} vertices\n'
                         '    RECEIVED: {} vertices'
                         .format(expected, len(data)))

    reduced = np.zeros((n_parc, data.shape[-1]), dtype=data.dtype)
    start = end = n_parc = 0
    for annot in [lhannot, rhannot]:
        # read files and update end index for `data`
        labels, ctab, names = read_annot(annot)
        names = _decode_list(names)

        indices = np.unique(labels)
        end += len(labels)

        for idx in range(data.shape[-1]):
            # get average of vertex-level data within parcels
            # set all NaN values to 0 before calling `_stats` because we are
            # returning sums, so the 0 values won't impact the sums (if we left
            # the NaNs then all parcels with even one NaN entry would be NaN)
            currdata = np.squeeze(data[start:end, idx])
            isna = np.isnan(currdata)
            counts, sums = _stats(np.nan_to_num(currdata), labels, indices)

            # however, we do need to account for the NaN values in the counts
            # so that our means are similar to what we'd get from e.g.,
            # np.nanmean here, our "sums" are the counts of NaN values in our
            # parcels
            _, nacounts = _stats(isna, labels, indices)
            counts = (np.asanyarray(counts, dtype=float)
                      - np.asanyarray(nacounts, dtype=float))

            with np.errstate(divide='ignore', invalid='ignore'):
                currdata = sums / counts

            # get indices of unkown and corpuscallosum and delete from parcels
            inds = sorted([names.index(f) for f in set(drop) & set(names)])
            currdata = np.delete(currdata, inds)

            # store parcellated data
            reduced[n_parc:n_parc + len(names) - len(inds), idx] = currdata
        start = end
        n_parc += len(names) - len(inds)

    return np.squeeze(reduced)


def _get_fsaverage_coords(version='fsaverage', surface='sphere'):
    """
    Gets vertex coordinates for specified `surface` of fsaverage `version`

    Parameters
    ----------
    version : str, optional
        One of {'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5',
        'fsaverage6'}. Default: 'fsaverage'
    surface : str, optional
        Surface for which to return vertex coordinates. Default: 'sphere'

    Returns
    -------
    coords : (N, 3) numpy.ndarray
        xyz coordinates of vertices for {left,right} hemisphere
    hemiid : (N,) numpy.ndarray
        Array denoting hemisphere designation of entries in `coords`, where
        `hemiid=0` denotes the left and `hemiid=1` the right hemisphere
    """

    # get coordinates and hemisphere designation for spin generation
    lhsphere, rhsphere = fetch_fsaverage(version)[surface]
    coords, hemi = [], []
    for n, sphere in enumerate([lhsphere, rhsphere]):
        coords.append(read_geometry(sphere)[0])
        hemi.append(np.ones(len(coords[-1])) * n)

    return np.row_stack(coords), np.hstack(hemi)


def spin_data(data, *, lhannot, rhannot, version='fsaverage', n_rotate=1000,
              drop=None, seed=None, verbose=False, return_cost=False):
    """
    Projects parcellated `data` to surface, rotates, and re-parcellates

    Projection to the surface uses `{lh,rh}annot` files. Rotation uses vertex
    coordinates from the specified fsaverage `version` and relies on
    :func:`netneurotools.stats.gen_spinsamples`. Re-parcellated data will not
    be exactly identical to original values due to re-averaging process.
    Parcels subsumed by regions in `drop` will be listed as NaN.

    Parameters
    ----------
    data : (N,) numpy.ndarray
        Parcellated data to be rotated. Parcels should be ordered by [left,
        right] hemisphere; ordering within hemisphere should correspond to the
        provided `{lh,rh}annot` annotation files.
    {lh,rh}annot : str
        Path to .annot file containing labels to parcels on the {left,right}
        hemisphere
    version : str, optional
        Specifies which version of `fsaverage` provided annotation files
        correspond to. Must be one of {'fsaverage', 'fsaverage3', 'fsaverage4',
        'fsaverage5', 'fsaverage6'}. Default: 'fsaverage'
    n_rotate : int, optional
        Number of rotations to generate. Default: 1000
    drop : list, optional
        Specifies regions in {lh,rh}annot that are not present in `data`. NaNs
        will be inserted in place of the these regions in the returned data. If
        not specified, 'unknown' and 'corpuscallosum' are assumed to not be
        present. Default: None
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None
    verbose : bool, optional
        Whether to print occasional status messages. Default: False
    return_cost : bool, optional
        Whether to return cost array (specified as Euclidean distance) for each
        coordinate for each rotation Default: True

    Returns
    -------
    rotated : (N, `n_rotate`) numpy.ndarray
        Rotated `data
    cost : (N, `n_rotate`,) numpy.ndarray
        Cost (specified as Euclidean distance) of re-assigning each coordinate
        for every rotation in `spinsamples`. Only provided if `return_cost` is
        True.
    """

    if drop is None:
        drop = [
            'unknown', 'corpuscallosum',  # default FreeSurfer
            'Background+FreeSurfer_Defined_Medial_Wall'  # common alternative
        ]

    # get coordinates and hemisphere designation for spin generation
    coords, hemiid = _get_fsaverage_coords(version, 'sphere')
    vertices = parcels_to_vertices(data, lhannot=lhannot, rhannot=rhannot,
                                   drop=drop)

    if len(vertices) != len(coords):
        raise ValueError('Provided annotation files have a different number '
                         'of vertices than the specified fsaverage surface.\n'
                         '    ANNOTATION: {} vertices\n'
                         '    FSAVERAGE:  {} vertices'
                         .format(len(vertices), len(coords)))

    spins, cost = gen_spinsamples(coords, hemiid, n_rotate=n_rotate,
                                  seed=seed, verbose=verbose)
    spun = np.zeros((len(data), n_rotate))
    for n in range(n_rotate):
        spun[:, n] = vertices_to_parcels(vertices[spins[:, n]],
                                         lhannot=lhannot, rhannot=rhannot,
                                         drop=drop)

    if return_cost:
        return spun, cost

    return spun


def spin_parcels(*, lhannot, rhannot, version='fsaverage', n_rotate=1000,
                 drop=None, seed=None, verbose=False, return_cost=False):
    """
    Rotates parcels in `{lh,rh}annot` and re-assigns based on maximum overlap

    Vertex labels are rotated with :func:`netneurotools.stats.gen_spinsamples`
    and a new label is assigned to each *parcel* based on the region maximally
    overlapping with its boundaries.

    Parameters
    ----------
    {lh,rh}annot : str
        Path to .annot file containing labels to parcels on the {left,right}
        hemisphere
    version : str, optional
        Specifies which version of `fsaverage` provided annotation files
        correspond to. Must be one of {'fsaverage', 'fsaverage3', 'fsaverage4',
        'fsaverage5', 'fsaverage6'}. Default: 'fsaverage'
    n_rotate : int, optional
        Number of rotations to generate. Default: 1000
    drop : list, optional
        Specifies regions in {lh,rh}annot that are not present in `data`. NaNs
        will be inserted in place of the these regions in the returned data. If
        not specified, 'unknown' and 'corpuscallosum' are assumed to not be
        present. Default: None
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None
    verbose : bool, optional
        Whether to print occasional status messages. Default: False
    return_cost : bool, optional
        Whether to return cost array (specified as Euclidean distance) for each
        coordinate for each rotation Default: True

    Returns
    -------
    spinsamples : (N, `n_rotate`) numpy.ndarray
        Resampling matrix to use in permuting data parcellated with labels from
        {lh,rh}annot, where `N` is the number of parcels. Indices of -1
        indicate that the parcel was completely encompassed by regions in
        `drop` and should be ignored.
    cost : (N, `n_rotate`,) numpy.ndarray
        Cost (specified as Euclidean distance) of re-assigning each coordinate
        for every rotation in `spinsamples`. Only provided if `return_cost` is
        True.
    """

    def overlap(vals):
        """ Returns most common non-negative value in `vals`; -1 if all neg
        """
        vals = np.asarray(vals)
        vals, counts = np.unique(vals[vals > 0], return_counts=True)
        try:
            return vals[counts.argmax()]
        except ValueError:
            return -1

    if drop is None:
        drop = [
            'unknown', 'corpuscallosum',  # default FreeSurfer
            'Background+FreeSurfer_Defined_Medial_Wall'  # common alternative
        ]
    drop = _decode_list(drop)

    # get vertex-level labels (set drop labels to - values)
    vertices, end = [], 0
    for n, annot in enumerate([lhannot, rhannot]):
        labels, ctab, names = read_annot(annot)
        names = _decode_list(names)
        todrop = set(names) & set(drop)
        inds = [names.index(f) - n for n, f in enumerate(todrop)]
        labs = np.arange(len(names) - len(inds)) + (end - (len(inds) * n))
        insert = np.arange(-1, -(len(inds) + 1), -1)
        vertices.append(np.insert(labs, inds, insert)[labels])
        end += len(names)
    vertices = np.hstack(vertices)
    labels = np.unique(vertices)
    mask = labels > -1

    # get coordinates and hemisphere designation for spin generation
    coords, hemiid = _get_fsaverage_coords(version, 'sphere')
    if len(vertices) != len(coords):
        raise ValueError('Provided annotation files have a different number '
                         'of vertices than the specified fsaverage surface.\n'
                         '    ANNOTATION: {} vertices\n'
                         '    FSAVERAGE:  {} vertices'
                         .format(len(vertices), len(coords)))

    # spin and assign regions based on max overlap
    spins, cost = gen_spinsamples(coords, hemiid, n_rotate=n_rotate,
                                  seed=seed, verbose=verbose)
    regions = np.zeros((len(labels[mask]), n_rotate), dtype='int32')
    for n in range(n_rotate):
        regions[:, n] = labeled_comprehension(vertices[spins[:, n]], vertices,
                                              labels, overlap, int, -1)[mask]

    if return_cost:
        return regions, cost

    return regions
