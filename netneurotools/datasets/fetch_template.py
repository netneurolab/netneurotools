"""Functions for fetching template data."""


import json
from pathlib import Path
import os.path as op

try:
    # nilearn 0.10.3
    from nilearn.datasets._utils import fetch_files
except ImportError:
    from nilearn.datasets.utils import _fetch_files as fetch_files

from sklearn.utils import Bunch

from .datasets_utils import (
    SURFACE,
    _get_data_dir, _get_dataset_info, _get_reference_info, _check_freesurfer_subjid
)


def fetch_fsaverage(
        version='fsaverage',
        data_dir=None, resume=True, verbose=1
    ):
    """
    Download files for fsaverage FreeSurfer template.

    This dataset contains

    If you used this data, please cite 1_, 2_, 3_.

    Parameters
    ----------
    version : str, optional
        One of {'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5',
        'fsaverage6'}. Default: 'fsaverage'

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['surf'] where corresponding values
        are length-2 lists downloaded template files (each list composed of
        files for the left and right hemisphere).

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

    References
    ----------
    .. [1] Anders M Dale, Bruce Fischl, and Martin I Sereno. Cortical
        surface-based analysis: i. segmentation and surface reconstruction.
        Neuroimage, 9(2):179\u2013194, 1999.
    .. [2] Bruce Fischl, Martin I Sereno, and Anders M Dale. Cortical
        surface-based analysis: ii: inflation, flattening, and a surface-based
        coordinate system. Neuroimage, 9(2):195\u2013207, 1999.
    .. [3] Bruce Fischl, Martin I Sereno, Roger BH Tootell, and Anders M Dale.
        High-resolution intersubject averaging and a coordinate system for the
        cortical surface. Human brain mapping, 8(4):272\u2013284, 1999.
    """
    versions = [
        'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6'
    ]
    if version not in versions:
        raise ValueError(
            f'The version of fsaverage requested {version} does not '
            f'exist. Must be one of {versions}'
        )

    dataset_name = 'tpl-fsaverage'
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ['orig', 'white', 'smoothwm', 'pial', 'inflated', 'sphere']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)[version]
    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': f'{dataset_name}.tar.gz'
    }

    _filenames = [
        f"{version}/surf/{hemi}.{surf}"
        for surf in keys for hemi in ['lh', 'rh']
    ]

    try:
        # use local FreeSurfer data if available
        data_dir = _check_freesurfer_subjid(version)[1]
        data = [op.join(data_dir, f) for f in _filenames]
    except FileNotFoundError:
        _filenames = [f"{dataset_name}/{_}" for _ in _filenames]
        _files = [(f, info['url'], opts) for f in _filenames]
        data = fetch_files(data_dir, files=_files, resume=resume, verbose=verbose)

    data = [SURFACE(*data[i:i + 2]) for i in range(0, len(keys) * 2, 2)]

    return Bunch(**dict(zip(keys, data)))


def fetch_hcp_standards(data_dir=None, resume=True, verbose=1):
    """
    Fetch HCP standard mesh atlases for converting between FreeSurfer and HCP.

    This dataset contains

    The original file was from 3_, but is no longer available. The archived
    file is available from 4_.

    If you used this data, please cite 1_, 2_.

    Returns
    -------
    standards : str
        Filepath to standard_mesh_atlases directory

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

    References
    ----------
    .. [1] David C Van Essen, Kamil Ugurbil, Edward Auerbach, Deanna
        Barch,Timothy EJ Behrens, Richard Bucholz, Acer Chang, Liyong Chen,
        Maurizio Corbetta, Sandra W Curtiss, and others. The human connectome
        project: a data acquisition perspective. Neuroimage,
        62(4):2222\u20132231, 2012.
    .. [2] Matthew F Glasser, Stamatios N Sotiropoulos, J Anthony Wilson,
        Timothy S Coalson, Bruce Fischl, Jesper L Andersson, Junqian Xu, Saad
        Jbabdi, Matthew Webster, Jonathan R Polimeni, and others. The minimal
        preprocessing pipelines for the human connectome project. Neuroimage,
        80:105\u2013124, 2013.
    .. [3] http://brainvis.wustl.edu/workbench/standard_mesh_atlases.zip
    .. [4] https://web.archive.org/web/20220121035833/http://brainvis.wustl.edu/workbench/standard_mesh_atlases.zip
    """
    dataset_name = 'tpl-hcp_standards'
    _get_reference_info(dataset_name, verbose=verbose)

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)["standard_mesh_atlases"]

    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': f'{dataset_name}.tar.gz'
    }
    fetched = fetch_files(
        data_dir,
        files=[(f'{dataset_name}/standard_mesh_atlases', info['url'], opts)],
        resume=resume, verbose=verbose
    )
    fetched = Path(fetched[0])

    return fetched


def fetch_civet(
        density='41k', version='v1',
        data_dir=None, resume=True, verbose=1
    ):
    """
    Fetch CIVET surface files.

    This dataset contains

    If you used this data, please cite 1_, 2_, 3_.

    Parameters
    ----------
    density : {'41k', '164k'}, optional
        Which density of the CIVET-space geometry files to fetch. The
        high-resolution '164k' surface only exists for version 'v2'
    version : {'v1, 'v2'}, optional
        Which version of the CIVET surfaces to use. Default: 'v2'

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['mid', 'white'] containing geometry
        files for CIVET surface. Note for version 'v1' the 'mid' and 'white'
        files are identical.

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
    License: https://github.com/aces/CIVET_Full_Project/blob/master/LICENSE

    References
    ----------
    .. [1] Oliver Lyttelton, Maxime Boucher, Steven Robbins, and Alan Evans. An
        unbiased iterative group registration template for cortical surface
        analysis. Neuroimage, 34(4):1535\u20131544, 2007.
    .. [2] Vladimir S Fonov, Alan C Evans, Robert C McKinstry, C Robert Almli,
        and DL Collins. Unbiased nonlinear average age-appropriate brain
        templates from birth to adulthood. NeuroImage, 47:S102, 2009.
    .. [3] Y Ad-Dab'bagh, O Lyttelton, J Muehlboeck, C Lepage, D Einarson, K
        Mok, O Ivanov, R Vincent, J Lerch, and E Fombonne. The civet
        image-processing environment: a fully automated comprehensive pipeline
        for anatomical neuroimaging research. proceedings of the 12th annual
        meeting of the organization for human brain mapping. Florence, Italy,
        pages 2266, 2006.
    """
    densities = ['41k', '164k']
    if density not in densities:
        raise ValueError(
            f'The density of CIVET requested "{density}" does not exist. '
            f'Must be one of {densities}'
        )
    versions = ['v1', 'v2']
    if version not in versions:
        raise ValueError(
            f'The version of CIVET requested "{version}" does not exist. '
            f'Must be one of {versions}'
        )

    if version == 'v1' and density == '164k':
        raise ValueError('The "164k" density CIVET surface only exists for '
                         'version "v2"')

    dataset_name = 'tpl-civet'
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ['mid', 'white']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)[version][f'civet{density}']

    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': f'{dataset_name}.tar.gz'
    }

    _filenames = [
        f"{dataset_name}/{version}/civet{density}/"
        f"tpl-civet_space-ICBM152_hemi-{hemi}_den-{density}_{surf}.obj"
        for surf in keys for hemi in ['L', 'R']
    ]
    _files = [(f, info['url'], opts) for f in _filenames]

    data = fetch_files(data_dir, files=_files, resume=resume, verbose=verbose)

    data = [SURFACE(*data[i:i + 2]) for i in range(0, len(keys) * 2, 2)]

    return Bunch(**dict(zip(keys, data)))


def fetch_conte69(data_dir=None, resume=True, verbose=1):
    """
    Download files for Van Essen et al., 2012 Conte69 template.

    This dataset contains

    If you used this data, please cite 1_, 2_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['midthickness', 'inflated',
        'vinflated'], where corresponding values are lists of filepaths to
        downloaded template files.

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

    References
    ----------
    .. [1] David C Van Essen, Kamil Ugurbil, Edward Auerbach, Deanna Barch,
        Timothy EJ Behrens, Richard Bucholz, Acer Chang, Liyong Chen, Maurizio
        Corbetta, Sandra W Curtiss, and others. The human connectome project: a
        data acquisition perspective. Neuroimage, 62(4):2222\u20132231, 2012.
    .. [2] David C Van Essen, Matthew F Glasser, Donna L Dierker, John Harwell,
        and Timothy Coalson. Parcellations and hemispheric asymmetries of human
        cerebral cortex analyzed on surface-based atlases. Cerebral cortex,
        22(10):2241\u20132262, 2012.
    .. [3] http://brainvis.wustl.edu/wiki/index.php//Caret:Atlases/Conte69_Atlas
    """
    dataset_name = 'tpl-conte69'
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ['midthickness', 'inflated', 'vinflated']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)
    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': f'{dataset_name}.tar.gz'
    }

    _filenames = [
        f"{dataset_name}/tpl-conte69_space-MNI305_variant-fsLR32k_{res}.{hemi}.surf.gii"
        for res in keys for hemi in ['L', 'R']
    ] + [
        f"{dataset_name}/template_description.json"
    ]
    _files = [(f, info['url'], opts) for f in _filenames]

    data = fetch_files(data_dir, files=_files, resume=resume, verbose=verbose)

    with open(data[-1], 'r') as src:
        data[-1] = json.load(src)

    # bundle hemispheres together
    data = [SURFACE(*data[:-1][i:i + 2]) for i in range(0, 6, 2)] + [data[-1]]

    return Bunch(**dict(zip(keys + ['info'], data)))


def fetch_yerkes19(data_dir=None, resume=None, verbose=1):
    """
    Download files for Donahue et al., 2016 Yerkes19 template.

    This dataset contains

    If you used this data, please cite 1_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['midthickness', 'inflated',
        'vinflated'], where corresponding values are lists of filepaths to
        downloaded template files.

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

    References
    ----------
    .. [1] Chad J Donahue, Stamatios N Sotiropoulos, Saad Jbabdi, Moises
        Hernandez-Fernandez, Timothy E Behrens, Tim B Dyrby, Timothy Coalson,
        Henry Kennedy, Kenneth Knoblauch, David C Van Essen, and others. Using
        diffusion tractography to predict cortical connection strength and
        distance: a quantitative comparison with tracers in the monkey. Journal
        of Neuroscience, 36(25):6758\u20136770, 2016.
    .. [2] https://balsa.wustl.edu/reference/show/976nz
    """
    dataset_name = 'tpl-yerkes19'
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ['midthickness', 'inflated', 'vinflated']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)
    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': f'{dataset_name}.tar.gz'
    }
    _filenames = [
        f"{dataset_name}/tpl-yerkes19_space-fsLR32k_{res}.{hemi}.surf.gii"
        for res in keys for hemi in ['L', 'R']

    ]
    _files = [(f, info['url'], opts) for f in _filenames]

    data = fetch_files(data_dir, files=_files, resume=resume, verbose=verbose)

    # bundle hemispheres together
    data = [SURFACE(*data[i:i + 2]) for i in range(0, 6, 2)]

    return Bunch(**dict(zip(keys + ['info'], data)))
