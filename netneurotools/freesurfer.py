# -*- coding: utf-8 -*-
"""
Functions for working with FreeSurfer data and parcellations
"""

import os
import os.path as op

import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist

from .datasets import fetch_fsaverage
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


def find_fsaverage_centroids(lhannot, rhannot, surf='sphere'):
    """
    Finds vertices corresponding to centroids of parcels in annotation files

    Note that using any other `surf` besides the default of 'sphere' may result
    in centroids that are not directly within the parcels themselves due to
    sulcal folding patterns.

    Parameters
    ----------
    {lh,rh}annot : str
        Path to .annot file containing labels to parcels on the {left,right}
        hemisphere
    surf : str, optional
        Surface on which to find parcel centroids. Default: 'sphere'

    Returns
    -------
    centroids : (N, 3) numpy.ndarray
        xyz coordinates of vertices closest to the centroid of each parcel
        defined in `lhannot` and `rhannot`
    hemiid : (N,) numpy.ndarray
        Array denoting hemisphere designation of coordinates in `centroids`,
        where `hemiid=0` denotes the left and `hemiid=1` the right hemisphere
    """

    surfaces = fetch_fsaverage()[surf]

    centroids, hemiid = [], []
    for n, (annot, surf) in enumerate(zip([lhannot, rhannot], surfaces)):
        vertices, faces = nib.freesurfer.read_geometry(surf)
        labels, ctab, names = nib.freesurfer.read_annot(annot)

        for lab in range(1, len(names)):
            if names[lab] == b'corpuscallosum':
                continue
            coords = np.atleast_2d(vertices[labels == lab].mean(axis=0))
            roi = vertices[np.argmin(cdist(vertices, coords), axis=0)[0]]
            centroids.append(roi)
            hemiid.append(n)

    return np.row_stack(centroids), np.asarray(hemiid)
