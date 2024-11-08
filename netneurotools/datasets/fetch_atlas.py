"""Functions for fetching atlas data."""
import itertools
import warnings

try:
    # nilearn 0.10.3
    from nilearn.datasets._utils import fetch_files
except ImportError:
    from nilearn.datasets.utils import _fetch_files as fetch_files

from sklearn.utils import Bunch

from .datasets_utils import (
    SURFACE,
    _get_data_dir, _get_dataset_info, _get_reference_info
)


def fetch_cammoun2012(
        version='MNI152NLin2009aSym',
        data_dir=None, resume=True, verbose=1
    ):
    """
    Download files for Cammoun et al., 2012 multiscale parcellation.

    This dataset contains

    If you used this data, please cite 1_.

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

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['scale033', 'scale060', 'scale125',
        'scale250', 'scale500'], where corresponding values are lists of
        filepaths to downloaded parcellation files.

    Other Parameters
    ----------------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    resume : bool, optional
        Whether to attempt to resume partial download, if possible. Default: True
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    License: https://raw.githubusercontent.com/LTS5/cmp/master/COPYRIGHT

    References
    ----------
    .. [1] Leila Cammoun, Xavier Gigandet, Djalel Meskaldji, Jean Philippe
        Thiran, Olaf Sporns, Kim Q Do, Philippe Maeder, Reto Meuli, and Patric
        Hagmann. Mapping the human connectome at multiple scales with diffusion
        spectrum mri. Journal of neuroscience methods, 203(2):386\u2013397,
        2012.
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
        raise ValueError(
            f'The version of Cammoun et al., 2012 parcellation '
            f'requested {version} does not exist. Must be one of {versions}'
        )

    dataset_name = 'atl-cammoun2012'
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ['scale033', 'scale060', 'scale125', 'scale250', 'scale500']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)[version]
    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': f'{dataset_name}.tar.gz'
    }

    # filenames differ based on selected version of dataset
    if version == 'MNI152NLin2009aSym':
        _filenames = [
            f'{dataset_name}/{version}/'
            f'atl-Cammoun2012_space-MNI152NLin2009aSym_res-{res[-3:]}'
            f'_deterministic{suff}'
            for res in keys for suff in ['.nii.gz']
        ] + [
            f'{dataset_name}/{version}/'
            f'atl-Cammoun2012_space-MNI152NLin2009aSym_info.csv'
        ]
    elif version == 'fslr32k':
        _filenames = [
            f'{dataset_name}/{version}/'
            f'atl-Cammoun2012_space-fslr32k_res-{res[-3:]}_hemi-{hemi}'
            f'_deterministic{suff}'
            for res in keys for hemi in ['L', 'R'] for suff in ['.label.gii']
        ]
    elif version in ('fsaverage', 'fsaverage5', 'fsaverage6'):
        _filenames = [
            f'{dataset_name}/{version}/'
            f'atl-Cammoun2012_space-{version}_res-{res[-3:]}_hemi-{hemi}'
            f'_deterministic{suff}'
            for res in keys for hemi in ['L', 'R'] for suff in ['.annot']
        ]
    else:
        _filenames = [
            f'{dataset_name}/{version}/'
            f'atl-Cammoun2012_res-{res[5:]}_hemi-{hemi}'
            f'_probabilistic{suff}'
            for res in keys[:-1] + ['scale500v1', 'scale500v2', 'scale500v3']
            for hemi in ['L', 'R'] for suff in ['.gcs', '.ctab']
        ]
    _files = [(f, info['url'], opts) for f in _filenames]
    data = fetch_files(data_dir, files=_files, resume=resume, verbose=verbose)

    if version == 'MNI152NLin2009aSym':
        keys += ['info']
    elif version in ('fslr32k', 'fsaverage', 'fsaverage5', 'fsaverage6'):
        data = [SURFACE(*data[i:i + 2]) for i in range(0, len(data), 2)]
    else:
        data = [data[::2][i:i + 2] for i in range(0, len(data) // 2, 2)]
        # deal with the fact that last scale is split into three files :sigh:
        data = data[:-3] + [list(itertools.chain.from_iterable(data[-3:]))]

    return Bunch(**dict(zip(keys, data)))


def fetch_schaefer2018(
        version='fsaverage',
        data_dir=None, resume=True, verbose=1
    ):
    """
    Download FreeSurfer .annot files for Schaefer et al., 2018 parcellation.

    This dataset contains

    If you used this data, please cite 1_.

    Parameters
    ----------
    version : {'fsaverage', 'fsaverage5', 'fsaverage6', 'fslr32k'}
        Specifies which surface annotation files should be matched to. Default:
        'fsaverage'

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys of format '{}Parcels{}Networks' where
        corresponding values are the left/right hemisphere annotation files

    Other Parameters
    ----------------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    resume : bool, optional
        Whether to attempt to resume partial download, if possible. Default: True
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    License: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

    References
    ----------
    .. [1] Alexander Schaefer, Ru Kong, Evan M Gordon, Timothy O Laumann,
        Xi-Nian Zuo, Avram J Holmes, Simon B Eickhoff, and BT Thomas Yeo.
        Local-global parcellation of the human cerebral cortex from intrinsic
        functional connectivity mri. Cerebral cortex, 28(9):3095\u20133114,
        2018.
    """
    versions = ['fsaverage', 'fsaverage5', 'fsaverage6', 'fslr32k']
    if version not in versions:
        raise ValueError(
            f'The version of Schaefer et al., 2018 parcellation '
            f'requested "{version}" does not exist. Must be one of {versions}'
        )

    dataset_name = 'atl-schaefer2018'
    _get_reference_info(dataset_name, verbose=verbose)

    keys = [
        f'{p}Parcels{n}Networks'
        for p in range(100, 1001, 100) for n in [7, 17]
    ]

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)[version]
    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': f'{dataset_name}.tar.gz'
    }

    if version == 'fslr32k':
        hemispheres, suffix = ['LR'], 'dlabel.nii'
    else:
        hemispheres, suffix = ['L', 'R'], 'annot'

    _filenames = [
        f'{dataset_name}/{version}/'
        f'atl-Schaefer2018_space-{version}_hemi-{hemi}_desc-{desc}'
        f'_deterministic.{suffix}'
        for desc in keys for hemi in hemispheres
    ]

    _files = [(f, info['url'], opts) for f in _filenames]

    data = fetch_files(data_dir, files=_files, resume=resume, verbose=verbose)

    if suffix == 'annot':
        data = [SURFACE(*data[i:i + 2]) for i in range(0, len(keys) * 2, 2)]

    return Bunch(**dict(zip(keys, data)))


def fetch_mmpall(
        version='fslr32k',
        data_dir=None, resume=True, verbose=1
    ):
    """
    Download .label.gii files for Glasser et al., 2016 MMPAll atlas.

    This dataset contains

    If you used this data, please cite 1_.

    Parameters
    ----------
    version : {'fslr32k'}
        Specifies which surface annotation files should be matched to. Default:
        'fslr32k'

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Namedtuple with fields ('lh', 'rh') corresponding to filepaths to
        left/right hemisphere parcellation files

    Other Parameters
    ----------------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    resume : bool, optional
        Whether to attempt to resume partial download, if possible. Default: True
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    License: https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms

    References
    ----------
    .. [1] Matthew F Glasser, Timothy S Coalson, Emma C Robinson, Carl D Hacker,
        John Harwell, Essa Yacoub, Kamil Ugurbil, Jesper Andersson, Christian F
        Beckmann, Mark Jenkinson, and others. A multi-modal parcellation of
        human cerebral cortex. Nature, 536(7615):171\u2013178, 2016.
    """
    versions = ['fslr32k']
    if version not in versions:
        raise ValueError(
            f'The version of Glasser et al., 2016 parcellation '
            f'requested "{version}" does not exist. Must be one of {versions}'
        )

    dataset_name = 'atl-mmpall'
    _get_reference_info(dataset_name, verbose=verbose)

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)[version]
    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': f'{dataset_name}.tar.gz'
    }

    _filenames = [
        f'{dataset_name}/{version}/'
        f'atl-MMPAll_space-{version}_hemi-{hemi}_deterministic.label.gii'
        for hemi in ['L', 'R']
    ]
    _files = [(f, info['url'], opts) for f in _filenames]

    data = fetch_files(data_dir, files=_files, resume=resume, verbose=verbose)

    return SURFACE(*data)


def fetch_pauli2018(data_dir=None, resume=True, verbose=1):
    """
    Download files for Pauli et al., 2018 subcortical parcellation.

    This dataset contains

    If you used this data, please cite 1_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['probabilistic', 'deterministic'],
        where corresponding values are filepaths to downloaded atlas files.

    Other Parameters
    ----------------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    resume : bool, optional
        Whether to attempt to resume partial download, if possible. Default: True
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    License: CC-BY Attribution 4.0 International

    References
    ----------
    .. [1] Wolfgang M Pauli, Amanda N Nili, and J Michael Tyszka. A
        high-resolution probabilistic in vivo atlas of human subcortical brain
        nuclei. Scientific data, 5(1):1\u201313, 2018.
    """
    dataset_name = 'atl-pauli2018'
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ['probabilistic', 'deterministic', 'info']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)

    _files = []
    for _, v in info.items():
        _f = f'{v["folder-name"]}/{v["file-name"]}'
        _url = v['url']
        _opts = {
            'md5sum': v['md5'],
            'move': f'{v["folder-name"]}/{v["file-name"]}'
        }
        _files.append(
            (_f, _url, _opts)
        )

    data = fetch_files(data_dir, files=_files, resume=resume, verbose=verbose)

    return Bunch(**dict(zip(keys, data)))


def fetch_ye2020():
    """Fetch Ye et al., 2020 subcortical parcellation."""
    pass


def fetch_voneconomo(data_dir=None, url=None, resume=True, verbose=1):
    """
    Fetch von-Economo Koskinas probabilistic FreeSurfer atlas.

    This dataset contains

    If you used this data, please cite 1_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['gcs', 'ctab', 'info']

    Other Parameters
    ----------------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    resume : bool, optional
        Whether to attempt to resume partial download, if possible. Default: True
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    License: CC-BY-NC-SA 4.0

    References
    ----------
    .. [1] Lianne H Scholtens, Marcel A de Reus, Siemon C de Lange, Ruben
        Schmidt, and Martijn P van den Heuvel. An mri von economo\u2013koskinas
        atlas. NeuroImage, 170:249\u2013256, 2018.
    """
    dataset_name = 'atl-voneconomo_koskinas'
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ['gcs', 'ctab', 'info']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)
    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': f'{dataset_name}.tar.gz'
    }

    _filenames = [
        f'{dataset_name}/'
        f'atl-vonEconomoKoskinas_hemi-{hemi}_probabilistic.{suff}'
        for hemi in ['L', 'R'] for suff in ['gcs', 'ctab']
    ] + [
        f'{dataset_name}/atl-vonEconomoKoskinas_info.csv'
    ]
    _files = [(f, info['url'], opts) for f in _filenames]
    data = fetch_files(data_dir, files=_files, resume=resume, verbose=verbose)

    data = [SURFACE(*data[:-1:2])] + [SURFACE(*data[1:-1:2])] + [data[-1]]

    return Bunch(**dict(zip(keys, data)))
