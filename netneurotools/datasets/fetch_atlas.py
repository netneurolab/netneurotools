"""Functions for fetching atlas data."""
import itertools
import os.path as op
import warnings

try:
    # nilearn 0.10.3
    from nilearn.datasets._utils import fetch_files
except ImportError:
    from nilearn.datasets.utils import _fetch_files as fetch_files

from sklearn.utils import Bunch

from .datasets_utils import (
    SURFACE,
    _get_data_dir, _get_dataset_info
)


def fetch_cammoun2012(version='MNI152NLin2009aSym', data_dir=None, url=None,
                      resume=True, verbose=1):
    """
    Download files for Cammoun et al., 2012 multiscale parcellation.

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
    data = fetch_files(data_dir, files=files, resume=resume, verbose=verbose)

    if version == 'MNI152NLin2009aSym':
        keys += ['info']
    elif version in ('fslr32k', 'fsaverage', 'fsaverage5', 'fsaverage6'):
        data = [SURFACE(*data[i:i + 2]) for i in range(0, len(data), 2)]
    else:
        data = [data[::2][i:i + 2] for i in range(0, len(data) // 2, 2)]
        # deal with the fact that last scale is split into three files :sigh:
        data = data[:-3] + [list(itertools.chain.from_iterable(data[-3:]))]

    return Bunch(**dict(zip(keys, data)))


def fetch_schaefer2018(version='fsaverage', data_dir=None, url=None,
                       resume=True, verbose=1):
    """
    Download FreeSurfer .annot files for Schaefer et al., 2018 parcellation.

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
    data = fetch_files(data_dir, files=files, resume=resume, verbose=verbose)

    if suffix == 'annot':
        data = [SURFACE(*data[i:i + 2]) for i in range(0, len(keys) * 2, 2)]

    return Bunch(**dict(zip(keys, data)))


def fetch_mmpall(version='fslr32k', data_dir=None, url=None, resume=True,
                 verbose=1):
    """
    Download .label.gii files for Glasser et al., 2016 MMPAll atlas.

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
    data = fetch_files(data_dir, files=files, resume=resume, verbose=verbose)

    return SURFACE(*data)


def fetch_pauli2018(data_dir=None, url=None, resume=True, verbose=1):
    """
    Download files for Pauli et al., 2018 subcortical parcellation.

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

    # format the query how fetch_files() wants things and then download data
    files = [
        (i['name'], i['url'], dict(md5sum=i['md5'], move=i['name']))
        for i in info
    ]

    data = fetch_files(data_dir, files=files, resume=resume, verbose=verbose)

    return Bunch(**dict(zip(keys, data)))


def fetch_ye2020():
    """Fetch Ye et al., 2020 subcortical parcellation."""
    pass


def fetch_voneconomo(data_dir=None, url=None, resume=True, verbose=1):
    """
    Fetch von-Economo Koskinas probabilistic FreeSurfer atlas.

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
    data = fetch_files(data_dir, files=files, resume=resume, verbose=verbose)
    data = [SURFACE(*data[:-1:2])] + [SURFACE(*data[1:-1:2])] + [data[-1]]

    return Bunch(**dict(zip(keys, data)))
