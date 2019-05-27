# -*- coding: utf-8 -*-
"""
Miscellaneous functions of various utility
"""

import glob
import os
import subprocess

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.utils.validation import check_array


def add_constant(data):
    """
    Adds a constant (i.e., intercept) term to `data`

    Parameters
    -----------
    data : (N, M) array_like
        Samples by features data array

    Returns
    -------
    data : (N, F) np.ndarray
        Where `F` is `M + 1`

    Examples
    --------
    >>> from netneurotools import utils

    >>> A = np.zeros((5, 5))
    >>> Ac = utils.add_constant(A)
    >>> Ac
    array([[0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.]])
    """

    data = check_array(data, ensure_2d=False)
    return np.column_stack([data, np.ones(len(data))])


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

    Examples
    --------
    >>> from netneurotools import utils

    >>> X = np.array([[1, 0.5, 0.25], [0.5, 1, 0.33], [0.25, 0.33, 1]])
    >>> tri = utils.get_triu(X)
    >>> tri
    array([0.5 , 0.25, 0.33])
    """

    return data[np.triu_indices(len(data), k=k)].copy()


def globpath(*args):
    """"
    Joins `args` with :py:func:`os.path.join` and returns sorted glob output

    Parameters
    ----------
    args : str
        Paths / `glob`-compatible regex strings

    Returns
    -------
    files : list
        Sorted list of files
    """

    return sorted(glob.glob(os.path.join(*args)))


def rescale(data, low=0, high=1):
    """
    Rescales `data` so it is within [`low`, `high`]

    Parameters
    ----------
    data : array_like
        Input data array
    low : float, optional
        Lower bound for rescaling. Default: -1
    high : float, optional
        Upper bound for rescaling. Default: 1

    Returns
    -------
    rescaled : np.ndarray
        Rescaled data
    """

    data = np.asarray(data)
    rescaled = np.interp(data, (data.min(), data.max()), (low, high))

    return rescaled


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

    Examples
    --------
    >>> from netneurotools import utils
    >>> p = utils.run('echo "hello world"', return_proc=True, quiet=True)
    >>> p.returncode
    0
    >>> p.stdout
    'hello world\\n'
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

    proc = subprocess.run(cmd, env=merged_env, shell=True, check=True,
                          universal_newlines=True, **opts)

    if return_proc:
        return proc


def check_fs_subjid(subject_id, subjects_dir=None):
    """
    Checks that `subject_id` exists in provided FreeSurfer `subjects_dir`

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
        subjects_dir = os.environ['SUBJECTS_DIR']
    else:
        subjects_dir = os.path.abspath(subjects_dir)

    subjdir = os.path.join(subjects_dir, subject_id)
    if not os.path.isdir(subjdir):
        raise FileNotFoundError('Cannot find specified subject id {} in '
                                'provided subject directory {}.'
                                .format(subject_id, subjects_dir))

    return subject_id, subjects_dir


def get_centroids(img, labels=None, image_space=False):
    """
    Finds centroids of `labels` in `img`

    Parameters
    ----------
    img : niimg-like object
        3D image containing integer label at each point
    labels : array_like, optional
        List of labels for which to find centroids. If not specified all
        labels present in `img` will be used. Zero will be ignored as it is
        considered "background." Default: None
    image_space : bool, optional
        Whether to return xyz (image space) coordinates for centroids based
        on transformation in `img.affine`. Default: False

    Returns
    -------
    centroids : (N, 3) np.ndarray
        Coordinates of centroids for ROIs in input data
    """

    from nilearn._utils import check_niimg_3d

    img = check_niimg_3d(img)
    data = img.get_data()

    if labels is None:
        labels = np.trim_zeros(np.unique(data))

    centroids = np.row_stack(ndimage.center_of_mass(data, labels=data,
                                                    index=labels))

    if image_space:
        centroids = nib.affines.apply_affine(img.affine, centroids)

    return centroids
