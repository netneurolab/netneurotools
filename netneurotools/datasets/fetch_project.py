"""Functions for fetching project data."""
import os
import os.path as op
import numpy as np

try:
    # nilearn 0.10.3
    from nilearn.datasets._utils import fetch_files as _fetch_files
except ImportError:
    from nilearn.datasets.utils import _fetch_files

from sklearn.utils import Bunch

from .datasets_utils import (
    _get_data_dir, _get_dataset_info
)

from ._mirchi2018 import _get_fc, _get_panas


def fetch_vazquez_rodriguez2019(data_dir=None, url=None, resume=True,
                                verbose=1):
    """
    Download files from Vazquez-Rodriguez et al., 2019, PNAS.

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


def fetch_mirchi2018(data_dir=None, resume=True, verbose=1):
    """
    Download (and creates) dataset for replicating Mirchi et al., 2018, SCAN.

    Parameters
    ----------
    data_dir : str, optional
        Directory to check for existing data files (if they exist) or to save
        generated data files. Files should be named mirchi2018_fc.npy and
        mirchi2018_panas.csv for the functional connectivity and behavioral
        data, respectively.

    Returns
    -------
    X : (73, 198135) numpy.ndarray
        Functional connections from MyConnectome rsfMRI time series data
    Y : (73, 13) numpy.ndarray
        PANAS subscales from MyConnectome behavioral data
    """
    data_dir = os.path.join(_get_data_dir(data_dir=data_dir), 'ds-mirchi2018')
    os.makedirs(data_dir, exist_ok=True)

    X_fname = os.path.join(data_dir, 'myconnectome_fc.npy')
    Y_fname = os.path.join(data_dir, 'myconnectome_panas.csv')

    if not os.path.exists(X_fname):
        X = _get_fc(data_dir=data_dir, resume=resume, verbose=verbose)
        np.save(X_fname, X, allow_pickle=False)
    else:
        X = np.load(X_fname, allow_pickle=False)

    if not os.path.exists(Y_fname):
        Y = _get_panas(data_dir=data_dir, resume=resume, verbose=verbose)
        np.savetxt(Y_fname, np.column_stack(list(Y.values())),
                   header=','.join(Y.keys()), delimiter=',', fmt='%i')
        # convert dictionary to structured array before returning
        Y = np.array([tuple(row) for row in np.column_stack(list(Y.values()))],
                     dtype=dict(names=list(Y.keys()), formats=['i8'] * len(Y)))
    else:
        Y = np.genfromtxt(Y_fname, delimiter=',', names=True, dtype=int)

    return X, Y


def fetch_hansen_manynetworks():
    """Download files from Hansen et al., 2023, PLOS Biology."""
    pass

def fetch_hansen_receptors():
    """Download files from Hansen et al., 2022, Nature Neuroscience."""
    pass

def fetch_hansen_genecognition():
    """Download files from Hansen et al., 2021, Nature Human Behaviour."""
    pass

def fetch_hansen_brainstem():
    """Download files from Hansen et al., 2024."""
    pass

def fetch_shafiei_hcpmeg():
    """Download files from Shafiei et al., 2022 & Shafiei et al., 2023."""
    pass

def fetch_suarez_mami():
    """Download files from Suarez et al., 2022, eLife."""
    pass



def fetch_famous_gmat(dataset, data_dir=None, url=None, resume=True,
                     verbose=1):
    """
    Download files from multi-species connectomes.

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
    available_connectomes = sorted(_get_dataset_info('ds-famous-gmat').keys())

    if dataset not in available_connectomes:
        raise ValueError('Provided dataset {} not available; must be one of {}'
                         .format(dataset, available_connectomes))

    dataset_name = 'ds-famous-gmat'

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


def fetch_neurosynth():
    """Download Neurosynth data."""
    pass
