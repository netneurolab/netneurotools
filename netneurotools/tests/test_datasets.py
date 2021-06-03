# -*- coding: utf-8 -*-
"""
For testing netneurotools.datasets functionality
"""

import os

import numpy as np
import pytest

from netneurotools import datasets
from netneurotools.datasets import utils


@pytest.mark.parametrize('corr, size, tol, seed', [
    (0.85, (1000,), 0.05, 1234),
    (0.85, (1000, 1000), 0.05, 1234),
    ([[1, 0.5, 0.3], [0.5, 1, 0], [0.3, 0, 1]], (1000,), 0.05, 1234)
])
def test_make_correlated_xy(corr, size, tol, seed):
    out = datasets.make_correlated_xy(corr=corr, size=size,
                                      tol=tol, seed=seed)
    # ensure output is expected shape
    assert out.shape[1:] == size
    assert len(out) == len(corr) if hasattr(corr, '__len__') else 2

    # check outputs are correlated within specified tolerance
    realcorr = np.corrcoef(out.reshape(len(out), -1))
    if len(realcorr) == 2 and not hasattr(corr, '__len__'):
        realcorr = realcorr[0, 1]
    assert np.all(np.abs(realcorr - corr) < tol)

    # check that seed generates reproducible values
    duplicate = datasets.make_correlated_xy(corr=corr, size=size,
                                            tol=tol, seed=seed)
    assert np.allclose(out, duplicate)


@pytest.mark.parametrize('corr', [
    (1.5), (-1.5),                                   # outside range of [-1, 1]
    ([0.85]), ([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),  # not 2D / square array
    ([[0.85]]), ([[1, 0.5], [0.5, 0.5]])             # diagonal not equal to 1
])
def test_make_correlated_xy_errors(corr):
    with pytest.raises(ValueError):
        datasets.make_correlated_xy(corr)


def test_fetch_conte69(tmpdir):
    conte = datasets.fetch_conte69(data_dir=tmpdir, verbose=0)
    assert all(hasattr(conte, k) for k in
               ['midthickness', 'inflated', 'vinflated', 'info'])


def test_fetch_yerkes19(tmpdir):
    conte = datasets.fetch_yerkes19(data_dir=tmpdir, verbose=0)
    assert all(hasattr(conte, k) for k in
               ['midthickness', 'inflated', 'vinflated'])


def test_fetch_pauli2018(tmpdir):
    pauli = datasets.fetch_pauli2018(data_dir=tmpdir, verbose=0)
    assert all(hasattr(pauli, k) and os.path.isfile(pauli[k]) for k in
               ['probabilistic', 'deterministic', 'info'])


@pytest.mark.parametrize('version', [
    'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6'
])
def test_fetch_fsaverage(tmpdir, version):
    fsaverage = datasets.fetch_fsaverage(version=version, data_dir=tmpdir,
                                         verbose=0)
    assert all(hasattr(fsaverage, k)
               and len(fsaverage[k]) == 2
               and all(os.path.isfile(hemi)
               for hemi in fsaverage[k]) for k in
               ['orig', 'white', 'smoothwm', 'pial', 'inflated', 'sphere'])


@pytest.mark.parametrize('version, expected', [
    ('MNI152NLin2009aSym', [1, 1, 1, 1, 1]),
    ('fsaverage', [2, 2, 2, 2, 2]),
    ('fsaverage5', [2, 2, 2, 2, 2]),
    ('fsaverage6', [2, 2, 2, 2, 2]),
    ('fslr32k', [2, 2, 2, 2, 2]),
    ('gcs', [2, 2, 2, 2, 6])
])
def test_fetch_cammoun2012(tmpdir, version, expected):
    keys = ['scale033', 'scale060', 'scale125', 'scale250', 'scale500']
    cammoun = datasets.fetch_cammoun2012(version, data_dir=tmpdir, verbose=0)

    # output has expected keys
    assert all(hasattr(cammoun, k) for k in keys)
    # and keys are expected lengths!
    for k, e in zip(keys, expected):
        out = getattr(cammoun, k)
        if isinstance(out, (tuple, list)):
            assert len(out) == e
        else:
            assert isinstance(out, str) and out.endswith('.nii.gz')

    if 'fsaverage' in version:
        with pytest.warns(DeprecationWarning):
            datasets.fetch_cammoun2012('surface', data_dir=tmpdir, verbose=0)


@pytest.mark.parametrize('dataset, expected', [
    ('celegans', ['conn', 'dist', 'labels', 'ref']),
    ('drosophila', ['conn', 'coords', 'labels', 'networks', 'ref']),
    ('human_func_scale033', ['conn', 'coords', 'labels', 'ref']),
    ('human_func_scale060', ['conn', 'coords', 'labels', 'ref']),
    ('human_func_scale125', ['conn', 'coords', 'labels', 'ref']),
    ('human_func_scale250', ['conn', 'coords', 'labels', 'ref']),
    ('human_func_scale500', ['conn', 'coords', 'labels', 'ref']),
    ('human_struct_scale033', ['conn', 'coords', 'dist', 'labels', 'ref']),
    ('human_struct_scale060', ['conn', 'coords', 'dist', 'labels', 'ref']),
    ('human_struct_scale125', ['conn', 'coords', 'dist', 'labels', 'ref']),
    ('human_struct_scale250', ['conn', 'coords', 'dist', 'labels', 'ref']),
    ('human_struct_scale500', ['conn', 'coords', 'dist', 'labels', 'ref']),
    ('macaque_markov', ['conn', 'dist', 'labels', 'ref']),
    ('macaque_modha', ['conn', 'coords', 'dist', 'labels', 'ref']),
    ('mouse', ['acronyms', 'conn', 'coords', 'dist', 'labels', 'ref']),
    ('rat', ['conn', 'labels', 'ref']),
])
def test_fetch_connectome(tmpdir, dataset, expected):
    connectome = datasets.fetch_connectome(dataset, data_dir=tmpdir, verbose=0)

    for key in expected:
        assert (key in connectome)
        assert isinstance(connectome[key], str if key == 'ref' else np.ndarray)


@pytest.mark.parametrize('version', [
    'fsaverage', 'fsaverage5', 'fsaverage6', 'fslr32k'
])
def test_fetch_schaefer2018(tmpdir, version):
    keys = [
        '{}Parcels{}Networks'.format(p, n)
        for p in range(100, 1001, 100) for n in [7, 17]
    ]
    schaefer = datasets.fetch_schaefer2018(version, data_dir=tmpdir, verbose=0)

    if version == 'fslr32k':
        assert all(k in schaefer and os.path.isfile(schaefer[k]) for k in keys)
    else:
        assert all(k in schaefer
                   and len(schaefer[k]) == 2
                   and all(os.path.isfile(hemi) for hemi in schaefer[k])
                   for k in keys)


def test_fetch_hcp_standards(tmpdir):
    hcp = datasets.fetch_hcp_standards(data_dir=tmpdir, verbose=0)
    assert os.path.isdir(hcp)


def test_fetch_mmpall(tmpdir):
    mmp = datasets.fetch_mmpall(data_dir=tmpdir, verbose=0)
    assert len(mmp) == 2
    assert all(os.path.isfile(hemi) for hemi in mmp)
    assert all(hasattr(mmp, attr) for attr in ('lh', 'rh'))


def test_fetch_voneconomo(tmpdir):
    vek = datasets.fetch_voneconomo(data_dir=tmpdir, verbose=0)
    assert all(hasattr(vek, k) and len(vek[k]) == 2 for k in ['gcs', 'ctab'])
    assert isinstance(vek.get('info'), str)


@pytest.mark.parametrize('dset, expected', [
    ('atl-cammoun2012', ['fsaverage', 'fsaverage5', 'fsaverage6', 'fslr32k',
                         'MNI152NLin2009aSym', 'gcs']),
    ('tpl-conte69', ['url', 'md5']),
    ('atl-pauli2018', ['url', 'md5', 'name']),
    ('tpl-fsaverage', ['fsaverage' + f for f in ['', '3', '4', '5', '6']]),
    ('atl-schaefer2018', ['fsaverage', 'fsaverage6', 'fsaverage6'])
])
def test_get_dataset_info(dset, expected):
    info = utils._get_dataset_info(dset)
    if isinstance(info, dict):
        assert all(k in info.keys() for k in expected)
    elif isinstance(info, list):
        for f in info:
            assert all(k in f.keys() for k in expected)
    else:
        assert False

    with pytest.raises(KeyError):
        utils._get_dataset_info('notvalid')


@pytest.mark.parametrize('version', [
    'v1', 'v2'
])
def test_fetch_civet(tmpdir, version):
    civet = datasets.fetch_civet(version=version, data_dir=tmpdir, verbose=0)
    for key in ('mid', 'white'):
        assert key in civet
        for hemi in ('lh', 'rh'):
            assert hasattr(civet[key], hemi)
            assert os.path.isfile(getattr(civet[key], hemi))


def test_get_data_dir(tmpdir):
    data_dir = utils._get_data_dir(tmpdir)
    assert os.path.isdir(data_dir)
