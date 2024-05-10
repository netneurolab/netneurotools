# -*- coding: utf-8 -*-
"""Utilites for loading / creating datasets."""

import json
import os
from collections import namedtuple
import importlib.resources

if getattr(importlib.resources, 'files', None) is not None:
    _importlib_avail = True
else:
    from pkg_resources import resource_filename
    _importlib_avail = False


SURFACE = namedtuple('Surface', ('lh', 'rh'))

FREESURFER_IGNORE = [
    'unknown', 'corpuscallosum', 'Background+FreeSurfer_Defined_Medial_Wall'
]

def _osfify_urls(data):
    """
    Format `data` object with OSF API URL.

    Parameters
    ----------
    data : object
        If dict with a `url` key, will format OSF_API with relevant values

    Returns
    -------
    data : object
        Input data with all `url` dict keys formatted
    """
    OSF_API = "https://files.osf.io/v1/resources/{}/providers/osfstorage/{}"

    if isinstance(data, str):
        return data
    elif 'url' in data:
        data['url'] = OSF_API.format(*data['url'])

    try:
        for key, value in data.items():
            data[key] = _osfify_urls(value)
    except AttributeError:
        for n, value in enumerate(data):
            data[n] = _osfify_urls(value)

    return data


if _importlib_avail:
    osf = importlib.resources.files("netneurotools") / "datasets/datasets.json"
else:
    osf = resource_filename('netneurotools', 'datasets/datasets.json')

with open(osf) as src:
    OSF_RESOURCES = _osfify_urls(json.load(src))


def _get_dataset_info(name):
    """
    Return url and MD5 checksum for dataset `name`.

    Parameters
    ----------
    name : str
        Name of dataset

    Returns
    -------
    url : str
        URL from which to download dataset
    md5 : str
        MD5 checksum for file downloade from `url`
    """
    try:
        return OSF_RESOURCES[name]
    except KeyError:
        raise KeyError("Provided dataset '{}' is not valid. Must be one of: {}"
                       .format(name, sorted(OSF_RESOURCES.keys()))) from None


def _get_data_dir(data_dir=None):
    """
    Get path to netneurotools data directory.

    Parameters
    ----------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None

    Returns
    -------
    data_dir : str
        Path to use as data directory
    """
    if data_dir is None:
        data_dir = os.environ.get('NNT_DATA', os.path.join('~', 'nnt-data'))
    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir


def _check_freesurfer_subjid(subject_id, subjects_dir=None):
    """
    Check that `subject_id` exists in provided FreeSurfer `subjects_dir`.

    Parameters
    ----------
    subject_id : str
        FreeSurfer subject ID
    subjects_dir : str, optional
        Path to FreeSurfer subject directory. If not set, will inherit from
        the environmental variable $SUBJECTS_DIR. Default: None

    Returns
    -------
    subject_id : str
        FreeSurfer subject ID, as provided
    subjects_dir : str
        Full filepath to `subjects_dir`

    Raises
    ------
    FileNotFoundError
    """
    # check inputs for subjects_dir and subject_id
    if subjects_dir is None or not os.path.isdir(subjects_dir):
        try:
            subjects_dir = os.environ['SUBJECTS_DIR']
        except KeyError:
            subjects_dir = os.getcwd()
    else:
        subjects_dir = os.path.abspath(subjects_dir)

    subjdir = os.path.join(subjects_dir, subject_id)
    if not os.path.isdir(subjdir):
        raise FileNotFoundError('Cannot find specified subject id {} in '
                                'provided subject directory {}.'
                                .format(subject_id, subjects_dir))

    return subject_id, subjects_dir


def _get_freesurfer_subjid(subject_id, subjects_dir=None):
    """
    Get fsaverage version `subject_id`, fetching if required.

    Parameters
    ----------
    subject_id : str
        FreeSurfer subject ID
    subjects_dir : str, optional
        Path to FreeSurfer subject directory. If not set, will inherit from
        the environmental variable $SUBJECTS_DIR. Default: None

    Returns
    -------
    subject_id : str
        FreeSurfer subject ID
    subjects_dir : str
        Path to subject directory with `subject_id`
    """

    # check for FreeSurfer install w/fsaverage; otherwise, fetch required
    try:
        subject_id, subjects_dir = _check_freesurfer_subjid(subject_id, subjects_dir)
    except FileNotFoundError:
        if 'fsaverage' not in subject_id:
            raise ValueError('Provided subject {} does not exist in provided '
                             'subjects_dir {}'
                             .format(subject_id, subjects_dir)) from None
        from ..datasets import fetch_fsaverage
        from ..datasets import _get_data_dir
        fetch_fsaverage(subject_id)
        subjects_dir = os.path.join(_get_data_dir(), 'tpl-fsaverage')
        subject_id, subjects_dir = _check_freesurfer_subjid(subject_id, subjects_dir)

    return subject_id, subjects_dir