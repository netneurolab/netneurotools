# -*- coding: utf-8 -*-
"""
Utilites for loading / creating datasets
"""

import os
import urllib


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


def _get_dataset(name, urls, data_dir=None, verbose=1):
    """
    Downloads dataset defined by `name`

    Parameters
    ----------
    name : str
        Name of dataset. Must be in :func:`pyls.examples.available_datasets()`
    data_dir : str
        Path to use as data directory to store dataset
    """

    data_dir = os.path.join(_get_data_dir(data_dir), name)
    os.makedirs(data_dir, exist_ok=True)

    for url in urls:
        parse = urllib.parse.urlparse(url)
        fname = os.path.join(data_dir, os.path.basename(parse.path))

        if not os.path.exists(fname):
            out = urllib.request.urlopen(url)
            with open(fname, 'wb') as dest:
                dest.write(out.read())
