"""Functions for fetching template data."""


import json
import os.path as op

try:
    # nilearn 0.10.3
    from nilearn.datasets._utils import fetch_files
except ImportError:
    from nilearn.datasets.utils import _fetch_files as fetch_files

from sklearn.utils import Bunch

from .datasets_utils import (
    SURFACE,
    _get_data_dir, _get_dataset_info, _check_freesurfer_subjid
)


def fetch_fsaverage(version='fsaverage', data_dir=None, url=None, resume=True,
                    verbose=1):
    """
    Download files for fsaverage FreeSurfer template.

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
        data_dir = _check_freesurfer_subjid(version)[1]
        data = [op.join(data_dir, f) for f in filenames]
    except FileNotFoundError:
        data = fetch_files(data_dir, resume=resume, verbose=verbose,
                            files=[(op.join(dataset_name, f), url, opts)
                                   for f in filenames])

    data = [SURFACE(*data[i:i + 2]) for i in range(0, len(keys) * 2, 2)]

    return Bunch(**dict(zip(keys, data)))


def fetch_hcp_standards(data_dir=None, url=None, resume=True, verbose=1):
    """
    Fetch HCP standard mesh atlases for converting between FreeSurfer and HCP.

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

    Notes
    -----
    Original file from: http://brainvis.wustl.edu/workbench/standard_mesh_atlases.zip
    Archived file from: https://web.archive.org/web/20220121035833/http://brainvis.wustl.edu/workbench/standard_mesh_atlases.zip
    """
    dataset_name = 'tpl-hcp_standards'
    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)["standard_mesh_atlases"]
    if url is None:
        url = info['url']

    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': '{}.tar.gz'.format(dataset_name)
    }
    filenames = [
        'L.sphere.32k_fs_LR.surf.gii', 'R.sphere.32k_fs_LR.surf.gii'
    ]
    files = [(op.join(dataset_name, "standard_mesh_atlases", f), url, opts)
             for f in filenames]

    fetch_files(data_dir, files=files, resume=resume, verbose=verbose)

    return op.join(data_dir, dataset_name)


def fetch_civet(density='41k', version='v1', data_dir=None, url=None,
                resume=True, verbose=1):
    """
    Fetch CIVET surface files.

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

    data = fetch_files(data_dir, resume=resume, verbose=verbose,
                        files=[(f, url, opts) for f in filenames])

    data = [SURFACE(*data[i:i + 2]) for i in range(0, len(keys) * 2, 2)]

    return Bunch(**dict(zip(keys, data)))


def fetch_conte69(data_dir=None, url=None, resume=True, verbose=1):
    """
    Download files for Van Essen et al., 2012 Conte69 template.

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

    data = fetch_files(data_dir, files=[(f, url, opts) for f in filenames],
                        resume=resume, verbose=verbose)

    with open(data[-1], 'r') as src:
        data[-1] = json.load(src)

    # bundle hemispheres together
    data = [SURFACE(*data[:-1][i:i + 2]) for i in range(0, 6, 2)] + [data[-1]]

    return Bunch(**dict(zip(keys + ['info'], data)))


def fetch_yerkes19(data_dir=None, url=None, resume=None, verbose=1):
    """
    Download files for Donahue et al., 2016 Yerkes19 template.

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

    data = fetch_files(data_dir, files=[(f, url, opts) for f in filenames],
                        resume=resume, verbose=verbose)

    # bundle hemispheres together
    data = [SURFACE(*data[i:i + 2]) for i in range(0, 6, 2)]

    return Bunch(**dict(zip(keys + ['info'], data)))
