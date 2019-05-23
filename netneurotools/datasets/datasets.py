# -*- coding: utf-8 -*-
"""
Dataset fetcher / creation / whathaveyou
"""

import itertools
import json
import os

from nilearn.datasets.utils import _fetch_files
import numpy as np
from sklearn.utils import Bunch
from sklearn.utils.validation import check_random_state

from .utils import _get_data_dir, _get_dataset_info


def fetch_cammoun2012(version='volume', data_dir=None, url=None, resume=True,
                      verbose=1):
    """
    Downloads files for Cammoun et al., 2012 multiscale parcellation

    Parameters
    ----------
    version : {'volume', 'surface', 'gcs'}
        Specifies which version of the dataset to download, where 'volume' will
        return .nii.gz atlas files defined in MNI152 space, 'surface' will
        return .annot files defined in fsaverage space (FreeSurfer 6.0.1), and
        'gcs' will return FreeSurfer-style .gcs probabilistic atlas files for
        generating new, subject-specific parcellations
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    url : str, optional
        URL from which to download data. Default: None
    resume : bool, optional
        Whether to attempt to resume partial download, if possible. Default:
        True
    verbose : int, optional
        Does nothing. Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Busnch`
        Dictionary-like object with keys ['scale033', 'scale060', 'scale125',
        'scale250', 'scale500'], where corresponding values are lists of
        filepaths to downloaded parcellation files.

    References
    ----------
    Cammoun, L., Gigandet, X., Meskaldji, D., Thiran, J. P., Sporns, O., Do, K.
    Q., Maeder, P., and Meuli, R., & Hagmann, P. (2012). Mapping the human
    connectome at multiple scales with diffusion spectrum MRI. Journal of
    Neuroscience Methods, 203(2), 386-397.

    Notes
    -----
    License: https://raw.githubusercontent.com/LTS5/cmp/master/COPYRIGHT
    """

    versions = ['volume', 'surface', 'gcs']
    if version not in versions:
        raise ValueError('The version of Cammoun et al., 2012 parcellation '
                         'requested "{}" does not exist. Must be one of {}'
                         .format(version, versions))

    dataset_name = 'atl-cammoun2012'
    keys = ['scale033', 'scale060', 'scale125', 'scale250', 'scale500']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)[version]
    if url is None:
        url = info['url']

    opts = {'uncompress': True, 'md5sum': info['md5'], 'move': 'tmp.tar.gz'}

    # filenames differ based on selected version of dataset
    if version == 'volume':
        filenames = [
            'atl-Cammoun2012_space-MNI152NLin2009aSym_res-{}_deterministic{}'
            .format(res[-3:], suff) for res in keys for suff in ['.nii.gz']
        ] + ['atl-Cammoun2012_space-MNI152NLin2009aSym_info.csv']
    elif version == 'surface':
        filenames = [
            'atl-Cammoun2012_space-fsaverage_res-{}_hemi-{}_deterministic{}'
            .format(res[-3:], hemi, suff) for res in keys
            for hemi in ['L', 'R'] for suff in ['.annot']
        ]
    else:
        filenames = [
            'atl-Cammoun2012_res-{}_hemi-{}_probabilistic{}'
            .format(res[5:], hemi, suff)
            for res in keys[:-1] + ['scale500v1', 'scale500v2', 'scale500v3']
            for hemi in ['L', 'R'] for suff in ['.gcs', '.ctab']
        ]

    files = [(os.path.join(dataset_name, f), url, opts) for f in filenames]
    data = _fetch_files(data_dir, files=files, resume=resume, verbose=verbose)

    if version == 'volume':
        keys += ['info']
    elif version == 'surface':
        data = [data[i:i + 2] for i in range(0, len(data), 2)]
    else:
        data = [data[::2][i:i + 2] for i in range(0, len(data) // 2, 2)]
        # deal with the fact that last scale is split into three files :sigh:
        data = data[:-3] + [list(itertools.chain.from_iterable(data[-3:]))]

    return Bunch(**dict(zip(keys, data)))


def fetch_conte69(data_dir=None, url=None, resume=True, verbose=1):
    """
    Downloads files for Van Essen et al., 2012 Conte69 template

    Parameters
    ----------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    url : str, optional
        URL from which to download data. Default: None
    resume : bool, optional
        Whether to attempt to resume partial download, if possible. Default:
        True
    verbose : int, optional
        Does nothing. Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Busnch`
        Dictionary-like object with keys ['midthickness', 'inflated',
        'vinflated'], where corresponding values are lists of filepaths to
        downloaded template files.

    References
    ----------
    http://brainvis.wustl.edu/wiki/index.php//Caret:Atlases/Conte69_Atlas

    Van Essen, D. C., Glasser, M. F., Dierker, D. L., Harwell, J., & Coalson,
    T. (2011). Parcellations and hemispheric asymmetries of human cerebral
    cortex analyzed on surface-based atlases. Cerebral cortex, 22(10),
    2241-2262.

    Notes
    -----
    License: ???
    """

    dataset_name = 'tpl-conte69'
    keys = ['midthickness', 'inflated', 'vinflated']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)
    if url is None:
        url = info['url']

    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': '{}.tar.gz'.format(dataset_name)
    }

    filenames = [
        'tpl-conte69/tpl-conte69_space-MNI305_variant-fsLR32k_{}.{}.surf.gii'
        .format(res, hemi) for res in keys for hemi in ['L', 'R']
    ] + ['tpl-conte69/template_description.json']

    data = _fetch_files(data_dir, files=[(f, url, opts) for f in filenames],
                        resume=resume, verbose=verbose)

    with open(data[-1], 'r') as src:
        data[-1] = json.load(src)

    # bundle hemispheres together
    data = [data[:-1][i:i + 2] for i in range(0, 6, 2)] + [data[-1]]

    return Bunch(**dict(zip(keys + ['info'], data)))


def fetch_pauli2018(data_dir=None, url=None, resume=True, verbose=1):
    """
    Downloads files for Pauli et al., 2018 subcortical parcellation

    Parameters
    ----------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    url : str, optional
        URL from which to download data. Default: None
    resume : bool, optional
        Whether to attempt to resume partial download, if possible. Default:
        True
    verbose : int, optional
        Does nothing. Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['probabilistic', 'deterministic'],
        where corresponding values are filepaths to downloaded atlas files.

    References
    ----------
    Pauli, W. M., Nili, A. N., & Tyszka, J. M. (2018). A high-resolution
    probabilistic in vivo atlas of human subcortical brain nuclei. Scientific
    Data, 5, 180063.

    Notes
    -----
    License: CC-BY Attribution 4.0 International
    """

    dataset_name = 'atl-pauli2018'
    keys = ['probabilistic', 'deterministic', 'info']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)

    # format the query how _fetch_files() wants things and then download data
    files = [
        (i['name'], i['url'], dict(md5sum=i['md5'], move=i['name']))
        for i in info
    ]

    data = _fetch_files(data_dir, files=files, resume=resume, verbose=verbose)

    return Bunch(**dict(zip(keys, data)))


def fetch_fsaverage(data_dir=None, url=None, resume=True, verbose=1):
    """
    Downloads files for fsaverage FreeSurfer template

    Parameters
    ----------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    url : str, optional
        URL from which to download data. Default: None
    resume : bool, optional
        Whether to attempt to resume partial download, if possible. Default:
        True
    verbose : int, optional
        Does nothing. Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['surf'] where corresponding values
        are length-2 lists downloaded template files (each list composed of
        files for the left and right hemisphere).

    References
    ----------

    """

    dataset_name = 'tpl-fsaverage'
    keys = ['orig', 'white', 'smoothwm', 'pial', 'inflated', 'sphere']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)
    if url is None:
        url = info['url']

    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': '{}.tar.gz'.format(dataset_name)
    }

    filenames = [
        'fsaverage/surf/{}.{}'
        .format(hemi, surf) for surf in keys for hemi in ['lh', 'rh']
    ]

    data = _fetch_files(data_dir, files=[(f, url, opts) for f in filenames],
                        resume=resume, verbose=verbose)
    data = [data[i:i + 2] for i in range(0, len(keys) * 2, 2)]

    return Bunch(**dict(zip(keys, data)))


def make_correlated_xy(corr=0.85, size=10000, seed=None, tol=0.001):
    """
    Generates random vectors that are correlated to approximately `corr`

    Parameters
    ----------
    corr : [-1, 1] float or (N, N) numpy.ndarray, optional
        The approximate correlation desired. If a float is provided, two
        vectors with the specified level of correlation will be generated. If
        an array is provided, it is assumed to be a symmetrical correlation
        matrix and ``len(corr)`` vectors with the specified levels of
        correlation will be generated. Default: 0.85
    size : int or tuple, optional
        Desired size of the generated vectors. Default: 1000
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None
    tol : [0, 1] float, optional
        Tolerance of correlation between generated `vectors` and specified
        `corr`. Default: 0.05

    Returns
    -------
    vectors : numpy.ndarray
        Random vectors of size `size` with correlation specified by `corr`

    Examples
    --------
    >>> from netneurotools import datasets

    By default two vectors are generated with specified correlation

    >>> x, y = datasets.make_correlated_xy()
    >>> np.corrcoef(x, y)  # doctest: +SKIP
    array([[1.        , 0.85083661],
           [0.85083661, 1.        ]])
    >>> x, y = datasets.make_correlated_xy(corr=0.2)
    >>> np.corrcoef(x, y)  # doctest: +SKIP
    array([[1.        , 0.20069953],
           [0.20069953, 1.        ]])

    You can also provide correlation matrices to generate more than two vectors
    if desired. Note that this makes it more difficult to ensure the actual
    correlations are close to the desired values:

    >>> corr = [[1, 0.5, 0.3], [0.5, 1, 0], [0.3, 0, 1]]
    >>> out = datasets.make_correlated_xy(corr=corr)
    >>> out.shape
    (3, 10000)
    >>> np.corrcoef(out)  # doctest: +SKIP
    array([[1.        , 0.50965273, 0.30235686],
           [0.50965273, 1.        , 0.01089107],
           [0.30235686, 0.01089107, 1.        ]])
    """

    rs = check_random_state(seed)

    # no correlations outside [-1, 1] bounds
    if np.any(np.abs(corr) > 1):
        raise ValueError('Provided `corr` must (all) be in range [-1, 1].')

    # if we're given a single number, assume two vectors are desired
    if isinstance(corr, (int, float)):
        covs = np.ones((2, 2)) * 0.111
        covs[(0, 1), (1, 0)] *= corr
    # if we're given a correlation matrix, assume `N` vectors are desired
    elif isinstance(corr, (list, np.ndarray)):
        corr = np.asarray(corr)
        if corr.ndim != 2 or len(corr) != len(corr.T):
            raise ValueError('If `corr` is a list or array, must be a 2D '
                             'square array, not {}'.format(corr.shape))
        if np.any(np.diag(corr) != 1):
            raise ValueError('Diagonal of `corr` must be 1.')
        covs = corr * 0.111
    means = [0] * len(covs)

    # generate the variables
    count = 0
    while count < 500:
        vectors = rs.multivariate_normal(mean=means, cov=covs, size=size).T
        flat = vectors.reshape(len(vectors), -1)
        # if diff between actual and desired correlations less than tol, break
        if np.all(np.abs(np.corrcoef(flat) - (covs / 0.111)) < tol):
            break
        count += 1

    return vectors
