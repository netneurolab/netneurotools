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


def test_fetch_pauli2018(tmpdir):
    pauli = datasets.fetch_pauli2018(data_dir=tmpdir, verbose=0)
    assert all(hasattr(pauli, k) and os.path.isfile(pauli[k]) for k in
               ['probabilistic', 'deterministic', 'info'])


def test_fetch_fsaverage(tmpdir):
    fsaverage = datasets.fetch_fsaverage(data_dir=tmpdir, verbose=0)
    assert all(hasattr(fsaverage, k)
               and len(fsaverage[k]) == 2
               and all(os.path.isfile(hemi)
               for hemi in fsaverage[k]) for k in
               ['orig', 'white', 'smoothwm', 'pial', 'inflated', 'sphere'])


@pytest.mark.parametrize('version, expected', [
    ('volume', [1, 1, 1, 1, 1]),
    ('surface', [2, 2, 2, 2, 2]),
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
        if isinstance(out, list):
            assert len(out) == e
        else:
            assert isinstance(out, str) and out.endswith('.nii.gz')


@pytest.mark.parametrize('dset, expected', [
    ('atl-cammoun2012', ['volume', 'surface', 'gcs']),
    ('tpl-conte69', ['url', 'md5']),
    ('atl-pauli2018', ['url', 'md5', 'name']),
    ('tpl-fsaverage', ['url', 'md5'])
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


def test_get_data_dir(tmpdir):
    data_dir = utils._get_data_dir(tmpdir)
    assert os.path.isdir(data_dir)
