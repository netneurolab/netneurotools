# -*- coding: utf-8 -*-
"""
Functions for fetching datasets from the internet
"""

from collections import namedtuple
import itertools
import json
import os.path as op
import warnings

from nilearn.datasets.utils import _fetch_files
import numpy as np
from sklearn.utils import Bunch

from .utils import _get_data_dir, _get_dataset_info
from ..utils import check_fs_subjid

SURFACE = namedtuple('Surface', ('lh', 'rh'))


def fetch_cammoun2012(version='MNI152NLin2009aSym', data_dir=None, url=None,
                      resume=True, verbose=1):
    """
    Downloads files for Cammoun et al., 2012 multiscale parcellation

    Parameters
    ----------
    version : str, optional
        Specifies which version of the dataset to download, where
        'MNI152NLin2009aSym' will return .nii.gz atlas files defined in MNI152
        space, 'fsaverageX' will return .annot files defined in fsaverageX
        space (FreeSurfer 6.0.1), 'fslr32k' will return .label.gii files in
        fs_LR_32k HCP space, and 'gcs' will return FreeSurfer-style .gcs
        probabilistic atlas files for generating new, subject-specific
        parcellations. Default: 'MNI152NLin2009aSym'
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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
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

    if version == 'surface':
        warnings.warn('Providing `version="surface"` is deprecated and will '
                      'be removed in a future release. For consistent '
                      'behavior please use `version="fsaverage"` instead.',
                      DeprecationWarning, stacklevel=2)
        version = 'fsaverage'
    elif version == 'volume':
        warnings.warn('Providing `version="volume"` is deprecated and will '
                      'be removed in a future release. For consistent '
                      'behavior please use `version="MNI152NLin2009aSym"` '
                      'instead.',
                      DeprecationWarning, stacklevel=2)
        version = 'MNI152NLin2009aSym'

    versions = [
        'gcs', 'fsaverage', 'fsaverage5', 'fsaverage6', 'fslr32k',
        'MNI152NLin2009aSym'
    ]
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

    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': '{}.tar.gz'.format(dataset_name)
    }

    # filenames differ based on selected version of dataset
    if version == 'MNI152NLin2009aSym':
        filenames = [
            'atl-Cammoun2012_space-MNI152NLin2009aSym_res-{}_deterministic{}'
            .format(res[-3:], suff) for res in keys for suff in ['.nii.gz']
        ] + ['atl-Cammoun2012_space-MNI152NLin2009aSym_info.csv']
    elif version == 'fslr32k':
        filenames = [
            'atl-Cammoun2012_space-fslr32k_res-{}_hemi-{}_deterministic{}'
            .format(res[-3:], hemi, suff) for res in keys
            for hemi in ['L', 'R'] for suff in ['.label.gii']
        ]
    elif version in ('fsaverage', 'fsaverage5', 'fsaverage6'):
        filenames = [
            'atl-Cammoun2012_space-{}_res-{}_hemi-{}_deterministic{}'
            .format(version, res[-3:], hemi, suff) for res in keys
            for hemi in ['L', 'R'] for suff in ['.annot']
        ]
    else:
        filenames = [
            'atl-Cammoun2012_res-{}_hemi-{}_probabilistic{}'
            .format(res[5:], hemi, suff)
            for res in keys[:-1] + ['scale500v1', 'scale500v2', 'scale500v3']
            for hemi in ['L', 'R'] for suff in ['.gcs', '.ctab']
        ]

    files = [
        (op.join(dataset_name, version, f), url, opts) for f in filenames
    ]
    data = _fetch_files(data_dir, files=files, resume=resume, verbose=verbose)

    if version == 'MNI152NLin2009aSym':
        keys += ['info']
    elif version in ('fslr32k', 'fsaverage', 'fsaverage5', 'fsaverage6'):
        data = [SURFACE(*data[i:i + 2]) for i in range(0, len(data), 2)]
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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
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
    data = [SURFACE(*data[:-1][i:i + 2]) for i in range(0, 6, 2)] + [data[-1]]

    return Bunch(**dict(zip(keys + ['info'], data)))


def fetch_yerkes19(data_dir=None, url=None, resume=None, verbose=1):
    """
    Downloads files for Donahue et al., 2016 Yerkes19 template

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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['midthickness', 'inflated',
        'vinflated'], where corresponding values are lists of filepaths to
        downloaded template files.

    References
    ----------
    https://balsa.wustl.edu/reference/show/976nz

    Donahue, C. J., Sotiropoulos, S. N., Jbabdi, S., Hernandez-Fernandez, M.,
    Behrens, T. E., Dyrby, T. B., ... & Glasser, M. F. (2016). Using diffusion
    tractography to predict cortical connection strength and distance: a
    quantitative comparison with tracers in the monkey. Journal of
    Neuroscience, 36(25), 6758-6770.

    Notes
    -----
    License: ???
    """

    dataset_name = 'tpl-yerkes19'
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
        'tpl-yerkes19/tpl-yerkes19_space-fsLR32k_{}.{}.surf.gii'
        .format(res, hemi) for res in keys for hemi in ['L', 'R']
    ]

    data = _fetch_files(data_dir, files=[(f, url, opts) for f in filenames],
                        resume=resume, verbose=verbose)

    # bundle hemispheres together
    data = [SURFACE(*data[i:i + 2]) for i in range(0, 6, 2)]

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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

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


def fetch_fsaverage(version='fsaverage', data_dir=None, url=None, resume=True,
                    verbose=1):
    """
    Downloads files for fsaverage FreeSurfer template

    Parameters
    ----------
    version : str, optional
        One of {'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5',
        'fsaverage6'}. Default: 'fsaverage'
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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['surf'] where corresponding values
        are length-2 lists downloaded template files (each list composed of
        files for the left and right hemisphere).

    References
    ----------

    """

    versions = [
        'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6'
    ]
    if version not in versions:
        raise ValueError('The version of fsaverage requested "{}" does not '
                         'exist. Must be one of {}'.format(version, versions))

    dataset_name = 'tpl-fsaverage'
    keys = ['orig', 'white', 'smoothwm', 'pial', 'inflated', 'sphere']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)[version]
    if url is None:
        url = info['url']

    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': '{}.tar.gz'.format(dataset_name)
    }

    filenames = [
        op.join(version, 'surf', '{}.{}'.format(hemi, surf))
        for surf in keys for hemi in ['lh', 'rh']
    ]

    try:
        data_dir = check_fs_subjid(version)[1]
        data = [op.join(data_dir, f) for f in filenames]
    except FileNotFoundError:
        data = _fetch_files(data_dir, resume=resume, verbose=verbose,
                            files=[(op.join(dataset_name, f), url, opts)
                                   for f in filenames])

    data = [SURFACE(*data[i:i + 2]) for i in range(0, len(keys) * 2, 2)]

    return Bunch(**dict(zip(keys, data)))


def available_connectomes():
    """
    Lists datasets available via :func:`~.fetch_connectome`

    Returns
    -------
    datasets : list of str
        List of available datasets
    """

    return sorted(_get_dataset_info('ds-connectomes').keys())


def fetch_connectome(dataset, data_dir=None, url=None, resume=True,
                     verbose=1):
    """
    Downloads files from multi-species connectomes

    Parameters
    ----------
    dataset : str
        Specifies which dataset to download; must be one of the datasets listed
        in :func:`netneurotools.datasets.available_connectomes()`.
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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object with, at a minimum, keys ['conn', 'labels',
        'ref'] providing connectivity / correlation matrix, region labels, and
        relevant reference. Other possible keys include 'dist' (an array of
        Euclidean distances between regions of 'conn'), 'coords' (an array of
        xyz coordinates for regions of 'conn'), 'acronyms' (an array of
        acronyms for regions of 'conn'), and 'networks' (an array of network
        affiliations for regions of 'conn')

    References
    ----------
    See `ref` key of returned dictionary object for relevant dataset reference
    """

    if dataset not in available_connectomes():
        raise ValueError('Provided dataset {} not available; must be one of {}'
                         .format(dataset, available_connectomes()))

    dataset_name = 'ds-connectomes'

    data_dir = op.join(_get_data_dir(data_dir=data_dir), dataset_name)
    info = _get_dataset_info(dataset_name)[dataset]
    if url is None:
        url = info['url']
    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': '{}.tar.gz'.format(dataset)
    }

    filenames = [
        op.join(dataset, '{}.csv'.format(fn)) for fn in info['keys']
    ] + [op.join(dataset, 'ref.txt')]
    data = _fetch_files(data_dir, files=[(f, url, opts) for f in filenames],
                        resume=resume, verbose=verbose)

    # load data
    for n, arr in enumerate(data[:-1]):
        try:
            data[n] = np.loadtxt(arr, delimiter=',')
        except ValueError:
            data[n] = np.loadtxt(arr, delimiter=',', dtype=str)
    with open(data[-1]) as src:
        data[-1] = src.read().strip()

    return Bunch(**dict(zip(info['keys'] + ['ref'], data)))


def fetch_vazquez_rodriguez2019(data_dir=None, url=None, resume=True,
                                verbose=1):
    """
    Downloads files from Vazquez-Rodriguez et al., 2019, PNAS

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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['rsquared', 'gradient'] containing
        1000 values from

    References
    ----------
    See `ref` key of returned dictionary object for relevant dataset reference
    """

    dataset_name = 'ds-vazquez_rodriguez2019'

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
        op.join(dataset_name, 'rsquared_gradient.csv')
    ]
    data = _fetch_files(data_dir, files=[(f, url, opts) for f in filenames],
                        resume=resume, verbose=verbose)

    # load data
    rsq, grad = np.loadtxt(data[0], delimiter=',', skiprows=1).T

    return Bunch(rsquared=rsq, gradient=grad)


def fetch_schaefer2018(version='fsaverage', data_dir=None, url=None,
                       resume=True, verbose=1):
    """
    Downloads FreeSurfer .annot files for Schaefer et al., 2018 parcellation

    Parameters
    ----------
    version : {'fsaverage', 'fsaverage5', 'fsaverage6', 'fslr32k'}
        Specifies which surface annotation files should be matched to. Default:
        'fsaverage'
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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys of format '{}Parcels{}Networks' where
        corresponding values are the left/right hemisphere annotation files

    References
    ----------
    Schaefer, A., Kong, R., Gordon, E. M., Laumann, T. O., Zuo, X. N., Holmes,
    A. J., ... & Yeo, B. T. (2017). Local-global parcellation of the human
    cerebral cortex from intrinsic functional connectivity MRI. Cerebral
    Cortex, 28(9), 3095-3114.

    Notes
    -----
    License: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    """

    versions = ['fsaverage', 'fsaverage5', 'fsaverage6', 'fslr32k']
    if version not in versions:
        raise ValueError('The version of Schaefer et al., 2018 parcellation '
                         'requested "{}" does not exist. Must be one of {}'
                         .format(version, versions))

    dataset_name = 'atl-schaefer2018'
    keys = [
        '{}Parcels{}Networks'.format(p, n)
        for p in range(100, 1001, 100) for n in [7, 17]
    ]

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)[version]
    if url is None:
        url = info['url']

    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': '{}.tar.gz'.format(dataset_name)
    }

    if version == 'fslr32k':
        hemispheres, suffix = ['LR'], 'dlabel.nii'
    else:
        hemispheres, suffix = ['L', 'R'], 'annot'
    filenames = [
        'atl-Schaefer2018_space-{}_hemi-{}_desc-{}_deterministic.{}'
        .format(version, hemi, desc, suffix)
        for desc in keys for hemi in hemispheres
    ]

    files = [(op.join(dataset_name, version, f), url, opts)
             for f in filenames]
    data = _fetch_files(data_dir, files=files, resume=resume, verbose=verbose)

    if suffix == 'annot':
        data = [SURFACE(*data[i:i + 2]) for i in range(0, len(keys) * 2, 2)]

    return Bunch(**dict(zip(keys, data)))


def fetch_hcp_standards(data_dir=None, url=None, resume=True, verbose=1):
    """
    Fetches HCP standard mesh atlases for converting between FreeSurfer and HCP

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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    standards : str
        Filepath to standard_mesh_atlases directory
    """
    if url is None:
        url = 'http://brainvis.wustl.edu/workbench/standard_mesh_atlases.zip'
    dataset_name = 'standard_mesh_atlases'
    data_dir = _get_data_dir(data_dir=data_dir)
    opts = {
        'uncompress': True,
        'move': '{}.zip'.format(dataset_name)
    }
    filenames = [
        'L.sphere.32k_fs_LR.surf.gii', 'R.sphere.32k_fs_LR.surf.gii'
    ]
    files = [(op.join(dataset_name, f), url, opts) for f in filenames]
    _fetch_files(data_dir, files=files, resume=resume, verbose=verbose)

    return op.join(data_dir, dataset_name)


def fetch_mmpall(version='fslr32k', data_dir=None, url=None, resume=True,
                 verbose=1):
    """
    Downloads .label.gii files for Glasser et al., 2016 MMPAll atlas

    Parameters
    ----------
    version : {'fslr32k'}
        Specifies which surface annotation files should be matched to. Default:
        'fslr32k'
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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Namedtuple with fields ('lh', 'rh') corresponding to filepaths to
        left/right hemisphere parcellation files

    References
    ----------
    Glasser, M. F., Coalson, T. S., Robinson, E. C., Hacker, C. D., Harwell,
    J., Yacoub, E., ... & Van Essen, D. C. (2016). A multi-modal parcellation
    of human cerebral cortex. Nature, 536(7615), 171-178.

    Notes
    -----
    License: https://www.humanconnectome.org/study/hcp-young-adult/document/
    wu-minn-hcp-consortium-open-access-data-use-terms
    """

    versions = ['fslr32k']
    if version not in versions:
        raise ValueError('The version of Glasser et al., 2016 parcellation '
                         'requested "{}" does not exist. Must be one of {}'
                         .format(version, versions))

    dataset_name = 'atl-mmpall'

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)[version]
    if url is None:
        url = info['url']
    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': '{}.tar.gz'.format(dataset_name)
    }

    hemispheres = ['L', 'R']
    filenames = [
        'atl-MMPAll_space-{}_hemi-{}_deterministic.label.gii'
        .format(version, hemi) for hemi in hemispheres
    ]

    files = [(op.join(dataset_name, version, f), url, opts) for f in filenames]
    data = _fetch_files(data_dir, files=files, resume=resume, verbose=verbose)

    return SURFACE(*data)


def fetch_voneconomo(data_dir=None, url=None, resume=True, verbose=1):
    """
    Fetches von-Economo Koskinas probabilistic FreeSurfer atlas

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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['gcs', 'ctab', 'info']

    References
    ----------
    Scholtens, L. H., de Reus, M. A., de Lange, S. C., Schmidt, R., & van den
    Heuvel, M. P. (2018). An MRI von Economo–Koskinas atlas. NeuroImage, 170,
    249-256.

    Notes
    -----
    License: CC-BY-NC-SA 4.0
    """

    dataset_name = 'atl-voneconomo_koskinas'
    keys = ['gcs', 'ctab', 'info']

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
        'atl-vonEconomoKoskinas_hemi-{}_probabilistic.{}'.format(hemi, suff)
        for hemi in ['L', 'R'] for suff in ['gcs', 'ctab']
    ] + ['atl-vonEconomoKoskinas_info.csv']
    files = [(op.join(dataset_name, f), url, opts) for f in filenames]
    data = _fetch_files(data_dir, files=files, resume=resume, verbose=verbose)
    data = [SURFACE(*data[:-1:2])] + [SURFACE(*data[1:-1:2])] + [data[-1]]

    return Bunch(**dict(zip(keys, data)))


def fetch_civet(density='41k', version='v1', data_dir=None, url=None,
                resume=True, verbose=1):
    """
    Fetches CIVET surface files

    Parameters
    ----------
    density : {'41k', '164k'}, optional
        Which density of the CIVET-space geometry files to fetch. The
        high-resolution '164k' surface only exists for version 'v2'
    version : {'v1, 'v2'}, optional
        Which version of the CIVET surfaces to use. Default: 'v2'
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
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['mid', 'white'] containing geometry
        files for CIVET surface. Note for version 'v1' the 'mid' and 'white'
        files are identical.

    References
    ----------
    Y. Ad-Dab’bagh, O. Lyttelton, J.-S. Muehlboeck, C. Lepage, D. Einarson, K.
    Mok, O. Ivanov, R. Vincent, J. Lerch, E. Fombonne, A. C. Evans, The CIVET
    image-processing environment: A fully automated comprehensive pipeline for
    anatomical neuroimaging research. Proceedings of the 12th Annual Meeting of
    the Organization for Human Brain Mapping (2006).

    Notes
    -----
    License: https://github.com/aces/CIVET_Full_Project/blob/master/LICENSE
    """

    densities = ['41k', '164k']
    if density not in densities:
        raise ValueError('The density of CIVET requested "{}" does not exist. '
                         'Must be one of {}'.format(density, densities))
    versions = ['v1', 'v2']
    if version not in versions:
        raise ValueError('The version of CIVET requested "{}" does not exist. '
                         'Must be one of {}'.format(version, versions))

    if version == 'v1' and density == '164k':
        raise ValueError('The "164k" density CIVET surface only exists for '
                         'version "v2"')

    dataset_name = 'tpl-civet'
    keys = ['mid', 'white']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)[version]['civet{}'.format(density)]
    if url is None:
        url = info['url']

    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': '{}.tar.gz'.format(dataset_name)
    }
    filenames = [
        op.join(dataset_name, version, 'civet{}'.format(density),
                'tpl-civet_space-ICBM152_hemi-{}_den-{}_{}.obj'
                .format(hemi, density, surf))
        for surf in keys for hemi in ['L', 'R']
    ]

    data = _fetch_files(data_dir, resume=resume, verbose=verbose,
                        files=[(f, url, opts) for f in filenames])

    data = [SURFACE(*data[i:i + 2]) for i in range(0, len(keys) * 2, 2)]

    return Bunch(**dict(zip(keys, data)))
