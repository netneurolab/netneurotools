# -*- coding: utf-8 -*-

import glob
import os
import os.path as op
import subprocess

import numpy as np
from sklearn.utils.validation import check_array


def globpath(*args):
    """"
    Joins provided `args` with os.path.join and then returns sorted glob

    Parameters
    ----------
    args : str
        Paths / glob-compatible regex

    Returns
    -------
    files : list
        Sorted list of files
    """

    return sorted(glob.glob(op.join(*args)))


def get_triu(data, k=1):
    """
    Returns vectorized version of upper triangle from `data`

    Parameters
    ----------
    data : (N, N) array_like
        Input data
    k : int, optional
        Which diagonal to select from (where primary diagonal is 0). Default: 1

    Returns
    -------
    triu : (N * N-1 / 2) numpy.ndarray
        Upper triangle of `data`
    """

    return data[np.triu_indices(len(data), k=1)].copy()


def add_constant(data):
    """
    Adds a constant (i.e., intercept) term to `data`

    Parameters
    -----------
    data : (N x M) array_like
        Samples by features data array

    Returns
    -------
    data : (N x F) np.ndarray
        Where ``F`` is ``M + 1``
    """

    data = check_array(data, ensure_2d=False)
    return np.column_stack([data, np.ones(len(data))])


def run(cmd, env=None, return_proc=False, quiet=False):
    """
    Runs `cmd` via shell subprocess with provided environment `env`

    Parameters
    ----------
    cmd : str
        Command to be run as single string
    env : dict, optional
        If provided, dictionary of key-value pairs to be added to base
        environment when running `cmd`. Default: None
    return_proc : bool, optional
        Whether to return CompletedProcess object. Default: false
    quiet : bool, optional
        Whether to suppress stdout/stderr from subprocess. Default: False

    Returns
    -------
    proc : subprocess.CompletedProcess
        Process output

    Raises
    ------
    subprocess.CalledProcessError
        If subprocess does not exit cleanly
    """

    merged_env = os.environ.copy()
    if env is not None:
        if not isinstance(env, dict):
            raise TypeError('Provided `env` must be a dictionary, not {}'
                            .format(type(env)))
        merged_env.update(env)

    opts = {}
    if quiet:
        opts = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        print('Running command: {}'.format(cmd))

    proc = subprocess.run(cmd, env=merged_env, shell=True, check=True,
                          universal_newlines=True, **opts)

    if return_proc:
        return proc
