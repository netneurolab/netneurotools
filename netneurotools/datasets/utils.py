# -*- coding: utf-8 -*-
"""
Utilites for loading / creating datasets
"""

import os


def _get_dataset_dir(name, data_dir=None):
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

    from nilearn.datasets.utils import _get_dataset_dir

    if data_dir is None:
        data_dir = os.environ.get('NNT_DATA', os.path.join('~', 'nnt-data'))
    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return _get_dataset_dir(name, data_dir=data_dir)
