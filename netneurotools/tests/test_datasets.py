# -*- coding: utf-8 -*-
"""
For testing netneurotools.datasets functionality
"""

import numpy as np
from netneurotools import datasets

import pytest


@pytest.mark.parametrize(('corr, size, tol, seed'), [
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


@pytest.mark.parametrize(('corr'), [
    (1.5), (-1.5),                                   # outside range of [-1, 1]
    ([0.85]), ([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),  # not 2D / square array
    ([[0.85]]), ([[1, 0.5], [0.5, 0.5]])             # diagonal not equal to 1
])
def test_make_correlated_xy_errors(corr):
    with pytest.raises(ValueError):
        datasets.make_correlated_xy(corr)
