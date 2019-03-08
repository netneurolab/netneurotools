# -*- coding: utf-8 -*-
"""
Utilites for loading / creating datasets
"""

import os

OSF_API = "https://files.osf.io/v1/resources/{}/providers/osfstorage/{}"
OSF_RESOURCES = {
    'atl-cammoun2012': {
        'url': OSF_API.format('mb37e', '5c59c82576653c001b27d7e4'),
        'md5': 'e5ff57e8970b13c5cd1b2348f70a685b'
    },
    'atl-pauli2018': [
        {
            'url': OSF_API.format('jkzwp', '5b11fa3364f25a001973dce0'),
            'md5': '62dd6ff405d3a8b89ee188cafa3a7f6a',
            'name': 'atl-pauli2018/atl-Pauli2018_space-MNI152NLin2009cAsym_hemi-both_probabilistic.nii.gz'  # noqa
        },
        {
            'url': OSF_API.format('jkzwp', '5b11fa2ff1f288000e625a7f'),
            'md5': '5a5b6246921be08456304875447c68ed',
            'name': 'atl-pauli2018/atl-Pauli2018_space-MNI152NLin2009cAsym_hemi-both_deterministic.nii.gz'  # noqa
        }
    ],
    # thanks to poldracklab team for uploading these useful datasets!
    'tpl-conte69': {
        'url': OSF_API.format('fvuh8', '5b198ec5ec24e20011b48548'),
        'md5': 'bd944e3f9f343e0e51e562b440960529'
    },
    'tpl-fsaverage': {
        'url': OSF_API.format('mb37e', '5c82830a1d73810018bdacea'),
        'md5': '3d2b99fd6c623e17ace6409c4207027a',
    }
}


def _get_dataset_info(name):
    """
    Returns url and MD5 checksum for dataset `name`

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
        raise KeyError('Provided dataset "{}" is not valid. Must be one of: {}'
                       .format(name, list(OSF_RESOURCES.keys())))


def _get_data_dir(data_dir=None):
    """
    Gets path to netneurotools data directory

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
