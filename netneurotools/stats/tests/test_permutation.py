"""For testing netneurotools.stats.permutation_test functionality."""

import pytest
import numpy as np
from netneurotools import stats

@pytest.mark.xfail
def test_permtest_1samp():
    """Test permutation test for one-sample t-test."""
    assert False
    # n1, n2, n3 = 10, 15, 20
    # rs = np.random.RandomState(1234)
    # rvn1 = rs.normal(loc=8, scale=10, size=(n1, n2, n3))

    # t1, p1 = stats.permtest_1samp(rvn1, 1, axis=0)


def test_permtest_rel():
    """Test permutation test for paired samples."""
    dr, pr = -0.0005, 0.4175824175824176
    dpr = ([dr, -dr], [pr, pr])

    rvs1 = np.linspace(1, 100, 100)
    rvs2 = np.linspace(1.01, 99.989, 100)
    rvs1_2D = np.array([rvs1, rvs2])
    rvs2_2D = np.array([rvs2, rvs1])

    # the p-values in these two cases should be consistent
    d, p = stats.permtest_rel(rvs1, rvs2, axis=0, seed=1234)
    assert np.allclose([d, p], (dr, pr))
    d, p = stats.permtest_rel(rvs1_2D.T, rvs2_2D.T, axis=0, seed=1234)
    assert np.allclose([d, p], dpr)

    # but the p-value will differ here because of _how_ we're drawing the
    # random permutations... it would be nice if this was consistent, but as
    # yet i don't have a great idea on how to make that happen without assuming
    # a whole lot about the data
    pr = 0.51248751
    tpr = ([dr, -dr], [pr, pr])
    d, p = stats.permtest_rel(rvs1_2D, rvs2_2D, axis=1, seed=1234)
    assert np.allclose([d, p], tpr)


def test_permtest_pearsonr():
    """Test permutation test for Pearson correlation."""
    np.random.seed(12345678)
    x, y = stats.make_correlated_xy(corr=0.1, size=100)
    r, p = stats.permtest_pearsonr(x, y)
    assert np.allclose([r, p], [0.10032564626876286, 0.3046953046953047])

    x, y = stats.make_correlated_xy(corr=0.5, size=100)
    r, p = stats.permtest_pearsonr(x, y)
    assert np.allclose([r, p], [0.500040365781984, 0.000999000999000999])

    z = x + np.random.normal(loc=1, size=100)
    r, p = stats.permtest_pearsonr(x, np.column_stack([y, z]))
    assert np.allclose(r, np.array([0.50004037, 0.25843187]))
    assert np.allclose(p, np.array([0.000999, 0.01098901]))

    a, b = stats.make_correlated_xy(corr=0.9, size=100)
    r, p = stats.permtest_pearsonr(np.column_stack([x, a]),
                                   np.column_stack([y, b]))
    assert np.allclose(r, np.array([0.50004037, 0.89927523]))
    assert np.allclose(p, np.array([0.000999, 0.000999]))

