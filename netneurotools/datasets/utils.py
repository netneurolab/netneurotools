# -*- coding: utf-8 -*-
"""
Utilites for loading / creating datasets
"""

import os

OSF_API = "https://files.osf.io/v1/resources/{}/providers/osfstorage/{}"
OSF_RESOURCES = {
    'atl-cammoun2012': (OSF_API.format('mb37e', '5c59c82576653c001b27d7e4'),
                        '14fa050e65e7e23e79a0a3f9f3e6c56d'),
    # thanks niworkflows team for uploading these useful datasets in one place!
    'tpl-conte69': (OSF_API.format('fvuh8', '5b198ec5ec24e20011b48548'),
                    'bd944e3f9f343e0e51e562b440960529')
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
