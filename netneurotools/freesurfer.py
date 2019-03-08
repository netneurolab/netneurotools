# -*- coding: utf-8 -*-
"""
Functions for working with FreeSurfer data and parcellations
"""
import os
import os.path as op
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
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

    cmd = 'mris_ca_label {opts}{subject_id} {hemi} {hemi}.sphere.reg {gcs} {annot}' # noqa

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


def combine_cammoun_500(files, subject_id, subjects_dir=None,
                        use_cache=True, quiet=False):
    """
    Combines finest parcellation from Cammoun et al., 2012 for `subject_id`

    The parcellations from Cammoun et al., 2012 have five distinct scales; the
    highest resolution parcellation (scale 500) is split into three GCS files
    for historical FreeSurfer purposes. This is a bit annoying for calculating
    statistics, plotting, etc., so this function can be run once all the GCS
    files have been used to produce annotations files for `subject_id` (using
    :py:func:`netneurotools.freesurfer.apply_parcellation`). This function will
    combine the three annot files that correspond to the highest resolution
    into a single annot file for a given subject

    Parameters
    ----------
    files : list of str
        List of filepaths to Cammoun et al., 2012 scale 500 parcellation
    subject_id : str
        FreeSurfer subject ID
    subjects_dir : str, optional
        Path to FreeSurfer subject directory. If not set, will inherit from
        the environmental variable `$SUBJECTS_DIR`. Default: None
    use_cache : bool, optional
        Whether to check for existence of relevant statistics file in directory
        specified by `{subjects_dir}/{subject_id}/stats' and use, if it exists.
        If False, will create a new stats file. Default: True
    quiet : bool, optional
        Whether to restrict status messages. Default: False

    Returns
    -------
    cammoun500 : list
        List of created annotation files
    """

    tolabel = 'mri_annotation2label --subject {subject_id} --hemi {hemi} --outdir {label_dir} --annotation {annot} --sd {subjects_dir}'  # noqa
    toannot = 'mris_label2annot --sd {subjects_dir} --s {subject_id} --ldir {label_dir} --hemi {hemi} --annot-path {annot} --ctab {ctab} {label}'  # noqa

    gcs_files = ['{hemi}' + f[3:] for f in files if f.startswith('rh')]
    subject_id, subjects_dir = check_fs_subjid(subject_id, subjects_dir)

    created = []
    for hemi in ['R', 'L']:
        # don't overwrite annotation file if it's already been created
        cammoun500 = os.path.join(subjects_dir, subject_id, 'label',
                                  '{}.cammoun500.annot'.format(hemi))
        if os.path.isfile(cammoun500) and use_cache:
            created.append(cammoun500)
            continue

        # make directory to temporarily store labels
        label_dir = os.path.join(subjects_dir, subject_id,
                                 '{}.cammoun500.labels'.format(hemi))
        os.makedirs(label_dir, exist_ok=True)

        ctab = pd.DataFrame(columns=range(5))
        for gcs in gcs_files:
            # grab relevant annotation file and convert to labels
            annot = apply_prob_atlas(subject_id, gcs.format(hemi=hemi),
                                     hemi=hemi, subjects_dir=subjects_dir)
            run(tolabel.format(subject_id=subject_id, hemi=hemi,
                               label_dir=label_dir, annot=annot,
                               subjects_dir=subjects_dir),
                quiet=quiet)

            # save ctab information from annotation file
            vtx, ct, names = nib.freesurfer.read_annot(annot)
            data = np.column_stack([[f.decode() for f in names], ct[:, :-1]])
            ctab = ctab.append(pd.DataFrame(data), ignore_index=True)

        # get rid of duplicate entries and add back in unknown/corpuscallosum
        ctab = ctab.drop_duplicates(subset=[0], keep=False)
        add_back = pd.DataFrame([['unknown', 25, 5, 25, 0],
                                 ['corpuscallosum', 120, 70, 50, 0]],
                                index=[0, 4])
        ctab = ctab.append(add_back).sort_index().reset_index(drop=True)
        # save ctab to temporary file for creation of annotation file
        ctab_fname = os.path.join(label_dir, '{}.cammoun500.ctab'.format(hemi))
        ctab.to_csv(ctab_fname, header=False, sep='\t', index=True)

        # get all labels EXCEPT FOR UNKNOWN to combine into annotation
        # unknown will be regenerated as all the unmapped vertices
        label = ' '.join(['--l {}'
                         .format(os.path.join(label_dir,
                                              '{hemi}.{lab}.label'
                                              .format(hemi=hemi, lab=lab)))
                          for lab in ctab.iloc[1:, 0]])
        # combine labels into annotation file
        run(toannot.format(subjects_dir=subjects_dir, subject_id=subject_id,
                           label_dir=label_dir, hemi=hemi, ctab=ctab_fname,
                           annot=cammoun500, label=label),
            quiet=quiet)
        created.append(cammoun500)

        # remove temporary label directory
        shutil.rmtree(label_dir)

    return created


def find_fsaverage_centroids(lhannot, rhannot, surf='sphere'):
    """
    Finds vertices corresponding to centroids of parcels in annotation files

    Note that using anything other than `surf=sphere` may result in centroids
    that are not directly within the parcels themselves due to sulcal folding.

    Parameters
    ----------
    lhannot : str
        Filepath to .annot file containing labels to parcels on the left
        hemisphere
    rhannot : str
        Filepath to .annot file containing labels to parcels on the right
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
