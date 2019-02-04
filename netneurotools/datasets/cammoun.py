# -*- coding: utf-8 -*-
"""
Loads parcellation information from Cammoun et al., 2012 (J Neurosci Methods)
"""

from itertools import chain
import os
import pickle
from pkg_resources import resource_filename
import shutil

import nibabel as nib
from nilearn.datasets.utils import _fetch_files
import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from .utils import _get_dataset_dir
from ..freesurfer import apply_parcellation
from ..utils import check_fs_subjid, run


def load_cammoun2012(scale, surface=True):
    """
    Returns centroids / hemi assignment of parcels from Cammoun et al., 2012

    Centroids are defined on the spherical projection of the fsaverage cortical
    surface reconstruciton (FreeSurfer v6.0.1)

    Parameters
    ----------
    scale : {33, 60, 125, 250, 500}
        Scale of parcellation for which to get centroids / hemisphere
        assignments

    Returns
    -------
    centroids : (N, 3) numpy.ndarray
        Centroids of parcels defined by Cammoun et al., 2012 parcellation
    hemiid : (N,) numpy.ndarray
        Hemisphere assignment of `centroids`, where 0 indicates left and 1
        indicates right hemisphere

    References
    ----------
    Cammoun, L., Gigandet, X., Meskaldji, D., Thiran, J. P., Sporns, O., Do, K.
    Q., Maeder, P., and Meuli, R., & Hagmann, P. (2012). Mapping the human
    connectome at multiple scales with diffusion spectrum MRI. Journal of
    Neuroscience Methods, 203(2), 386-397.

    Examples
    --------
    >>> from netneurotools import datasets

    >>> coords, hemiid = datasets.load_cammoun2012(scale=33)
    >>> coords.shape, hemiid.shape
    ((68, 3), (68,))

    ``hemiid`` is a vector of 0 and 1 denoting which ``coords`` are in the
    left / right hemisphere, respectively:

    >>> hemiid
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1])
    """

    pckl = resource_filename('netneurotools', 'data/cammoun.pckl')

    if not isinstance(scale, int):
        try:
            scale = int(scale)
        except ValueError:
            raise ValueError('Provided `scale` must be integer in [33, 60, '
                             '125, 250, 500], not {}'.format(scale))
    if scale not in [33, 60, 125, 250, 500]:
        raise ValueError('Provided `scale` must be integer in [33, 60, 125, '
                         '250, 500], not {}'.format(scale))

    with open(pckl, 'rb') as src:
        data = pickle.load(src)['cammoun{}'.format(str(scale))]

    return Bunch(**data)


def fetch_cammoun2012(ctab=False, data_dir=None, url=None, resume=True,
                      verbose=1):
    """
    Downloads files for Cammoun et al., 2012 multiscale parcellation

    Parcellation is comprised of FreeSurfer ".gcs" files; consider using
    :py:func:`netneurotools.freesurfer.apply_parcellation` to map them to a
    FreeSurfer-processed subject

    Parameters
    ----------
    ctab: bool, optional
        Whether to return colortable files corresponding to parcellation.
        Default: True
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    url : str, optional
        URL where parcellation files are maintained. Default:
    resume : bool, optional
        Whether to attempt to resume partial download, if possible. Default:
        True
    verbose : int, optional
        Does nothing. Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['scale36', 'scale60', 'scale125',
        'scale250', 'scale500'], where corresponding values are lists of
        filepaths to downloaded parcellation files. If `ctab=True`, values are
        lists of tuples of (parcellation, colortable).

    References
    ----------
    Cammoun, L., Gigandet, X., Meskaldji, D., Thiran, J. P., Sporns, O., Do, K.
    Q., Maeder, P., and Meuli, R., & Hagmann, P. (2012). Mapping the human
    connectome at multiple scales with diffusion spectrum MRI. Journal of
    Neuroscience Methods, 203(2), 386-397.
    """

    if url is None:
        url = ("https://raw.githubusercontent.com/LTS5/cmp/master/cmp/data"
               "/colortable_and_gcs/{fpath}")
    dataset_name = 'cammoun2012'
    keys = ['scale36', 'scale60', 'scale125', 'scale250', 'scale500']

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)

    filenames = []
    for hemi in ['rh', 'lh']:
        h = hemi[0].capitalize()  # hemi designations differ b/w gcs and ctab
        rename = f'{hemi}.cammoun'
        filenames += [
            # alternating GCS, CTAB file
            (f'my_atlas_gcs/myatlas_36_{hemi}.gcs', f'{rename}36.gcs'),
            (f'original_color_36_{h}.txt', f'{rename}36.ctab'),
            (f'my_atlas_gcs/myatlas_60_{hemi}.gcs', f'{rename}60.gcs'),
            (f'original_color_60_{h}.txt', f'{rename}60.ctab'),
            (f'my_atlas_gcs/myatlas_125_{hemi}.gcs', f'{rename}125.gcs'),
            (f'original_color_125_{h}.txt', f'{rename}125.ctab'),
            (f'my_atlas_gcs/myatlas_250_{hemi}.gcs', f'{rename}250.gcs'),
            (f'original_color_250_{h}.txt', f'{rename}250.ctab'),
            (f'my_atlas_gcs/myatlasP1_16_{hemi}.gcs', f'{rename}P1_16.gcs'),
            (f'original_color_P1_16_{h}.txt', f'{rename}P1_16.ctab'),
            (f'my_atlas_gcs/myatlasP17_28_{hemi}.gcs', f'{rename}P17_28.gcs'),
            (f'original_color_P17_28_{h}.txt', f'{rename}P17_28.ctab'),
            (f'my_atlas_gcs/myatlasP29_36_{hemi}.gcs', f'{rename}P29_36.gcs'),
            (f'original_color_P29_36_{h}.txt', f'{rename}P29_36.ctab')
        ]
    if not ctab:  # drop colortable files if we don't want to download them
        filenames = filenames[::2]

    filenames = [
        (os.path.basename(out),                   # expected output filename
         url.format(fpath=loc),                   # url of file
         dict(move=os.path.join(data_dir, out)))  # rename file after download
        for (loc, out) in filenames
    ]
    data = _fetch_files(data_dir, filenames, resume=resume, verbose=verbose)

    # join gcs and colortable files into tuple, if downloaded
    if ctab:
        data = [tuple(data[i:(i + 2)]) for i in range(0, len(data), 2)]

    # join hemispheres for each scale
    scales = [data[n::7] for n in range(7)]

    # deal with the fact that the last scale is split into three files :sigh:
    scales = scales[:-3] + [list(chain.from_iterable(scales[-3:]))]

    return Bunch(**dict(zip(keys, scales)))


def combine_cammoun_500(filepath, subject_id, subjects_dir=None,
                        use_cache=True, quiet=False):
    """
    Combines highest scale parcellation from Cammoun 2012 for `subject_id`

    The parcellations from Cammoun, 2012 have five distinct scales; the highest
    resolution parcellation (scale 500) is split into three GCS files for what
    are likely historical purposes. This is a bit annoying for calculating
    statistics, plotting, etc., so this function can be run once all the GCS
    files have been used to produce annotations files for `subject_id` (using
    :py:func:`netneurotools.freesurfer.apply_parcellation`). This function will
    combine the three files that correspond to the highest resolution scale
    into a single annot file for a given subject, and then run `get_statistics`
    on that file to cache it

    Parameters
    ----------
    filepath : str
        Path to directory containing all Cammoun scale500 .gcs files
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

    gcs_files = [os.path.join(filepath, gcs) for gcs in [
        'myatlasP1_16_{hemi}.gcs',
        'myatlasP17_28_{hemi}.gcs',
        'myatlasP29_36_{hemi}.gcs'
    ]]
    subject_id, subjects_dir = check_fs_subjid(subject_id, subjects_dir)

    created = []
    for hemi in ['lh', 'rh']:
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
            annot = apply_parcellation(subject_id, gcs.format(hemi=hemi),
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
