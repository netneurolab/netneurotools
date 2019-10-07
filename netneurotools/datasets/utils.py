# -*- coding: utf-8 -*-
"""
Utilites for loading / creating datasets
"""

import json
import os
from pkg_resources import resource_filename


def _osfify_urls(data):
    """
    Formats `data` object with OSF API URL

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


with open(resource_filename('netneurotools', 'data/osf.json')) as src:
    OSF_RESOURCES = _osfify_urls(json.load(src))


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
        raise KeyError("Provided dataset '{}' is not valid. Must be one of: {}"
                       .format(name, sorted(OSF_RESOURCES.keys())))


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
