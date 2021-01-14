# -*- coding: utf-8 -*-
"""
For testing netneurotools.stats functionality
"""

import itertools
import numpy as np
import pytest

from netneurotools import datasets, stats


@pytest.mark.xfail
def test_permtest_1samp():
    assert False
    # n1, n2, n3 = 10, 15, 20
    # rs = np.random.RandomState(1234)
    # rvn1 = rs.normal(loc=8, scale=10, size=(n1, n2, n3))

    # t1, p1 = stats.permtest_1samp(rvn1, 1, axis=0)


def test_permtest_rel():
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
    np.random.seed(12345678)
    x, y = datasets.make_correlated_xy(corr=0.1, size=100)
    r, p = stats.permtest_pearsonr(x, y)
    assert np.allclose([r, p], [0.10032564626876286, 0.3046953046953047])

    x, y = datasets.make_correlated_xy(corr=0.5, size=100)
    r, p = stats.permtest_pearsonr(x, y)
    assert np.allclose([r, p], [0.500040365781984, 0.000999000999000999])

    z = x + np.random.normal(loc=1, size=100)
    r, p = stats.permtest_pearsonr(x, np.column_stack([y, z]))
    assert np.allclose(r, np.array([0.50004037, 0.25843187]))
    assert np.allclose(p, np.array([0.000999, 0.01098901]))

    a, b = datasets.make_correlated_xy(corr=0.9, size=100)
    r, p = stats.permtest_pearsonr(np.column_stack([x, a]),
                                   np.column_stack([y, b]))
    assert np.allclose(r, np.array([0.50004037, 0.89927523]))
    assert np.allclose(p, np.array([0.000999, 0.000999]))


@pytest.mark.parametrize('x, y, expected', [
    # basic one-dimensional input
    (range(5), range(5), (1.0, 0.0)),
    # broadcasting occurs regardless of input order
    (np.stack([range(5), range(5, 0, -1)], 1), range(5),
     ([1.0, -1.0], [0.0, 0.0])),
    (range(5), np.stack([range(5), range(5, 0, -1)], 1),
     ([1.0, -1.0], [0.0, 0.0])),
    # correlation between matching columns
    (np.stack([range(5), range(5, 0, -1)], 1),
     np.stack([range(5), range(5, 0, -1)], 1),
     ([1.0, 1.0], [0.0, 0.0]))
])
def test_efficient_pearsonr(x, y, expected):
    assert np.allclose(stats.efficient_pearsonr(x, y), expected)


def test_efficient_pearsonr_errors():
    with pytest.raises(ValueError):
        stats.efficient_pearsonr(range(4), range(5))

    assert all(np.isnan(a) for a in stats.efficient_pearsonr([], []))


def test_gen_rotation():
    # make a few rotations (some same / different)
    rout1, lout1 = stats._gen_rotation(seed=1234)
    rout2, lout2 = stats._gen_rotation(seed=1234)
    rout3, lout3 = stats._gen_rotation(seed=5678)

    # confirm consistency with the same seed
    assert np.allclose(rout1, rout2) and np.allclose(lout1, lout2)

    # confirm inconsistency with different seeds
    assert not np.allclose(rout1, rout3) and not np.allclose(lout1, lout3)

    # confirm reflection across L/R hemispheres as expected
    # also confirm min/max never exceeds -1/1
    reflected = np.array([[1, -1, -1], [-1, 1, 1], [-1, 1, 1]])
    for r, l in zip([rout1, rout3], [lout1, lout3]):
        assert np.allclose(r / l, reflected)
        assert r.max() < 1 and r.min() > -1 and l.max() < 1 and l.min() > -1


def _get_sphere_coords(s, t, r=1):
    """ Gets coordinates at angles `s` and `t` a sphere of radius `r`
    """
    # convert to radians
    rad = np.pi / 180
    s, t = s * rad, t * rad

    # calculate new points
    x = r * np.cos(s) * np.sin(t)
    y = r * np.sin(s) * np.cos(t)
    z = r * np.cos(t)

    return x, y, z


def test_gen_spinsamples():
    # grab a few points from a spherical surface and duplicate it for the
    # "other hemisphere"
    coords = [_get_sphere_coords(s, t, r=1) for s, t in
              itertools.product(range(0, 360, 45), range(0, 360, 45))]
    coords = np.row_stack([coords, coords])
    hemi = np.hstack([np.zeros(len(coords) // 2), np.ones(len(coords) // 2)])

    # generate "normal" test spins
    spins, cost = stats.gen_spinsamples(coords, hemi, n_rotate=10, seed=1234,
                                        return_cost=True)
    assert spins.shape == spins.shape == (len(coords), 10)

    # confirm that `method` parameter functions as desired
    for method in ['vasa', 'hungarian']:
        spin_exact, cost_exact = stats.gen_spinsamples(coords, hemi,
                                                       n_rotate=10, seed=1234,
                                                       method=method,
                                                       return_cost=True)
        assert spin_exact.shape == cost.shape == (len(coords), 10)
        for s in spin_exact.T:
            assert len(np.unique(s)) == len(s)

    # check that one hemisphere works
    mask = hemi == 0
    spins, cost = stats.gen_spinsamples(coords[mask], hemi[mask], n_rotate=10,
                                        seed=1234, return_cost=True)
    assert spins.shape == cost.shape == (len(coords[mask]), 10)

    # confirm that check_duplicates will raise warnings
    # since spins aren't exact permutations we need to use 4C4 with repeats
    # and then perform one more rotation than that number (i.e., 35 + 1)
    with pytest.warns(UserWarning):
        i = [0, 1, -2, -1]  # only grab a few coordinates
        stats.gen_spinsamples(coords[i], hemi[i], n_rotate=36, seed=1234)

    # non-3D coords
    with pytest.raises(ValueError):
        stats.gen_spinsamples(coords[:, :2], hemi)

    # non-1D hemi
    with pytest.raises(ValueError):
        stats.gen_spinsamples(coords, np.column_stack([hemi, hemi]))

    # different length coords and hemi
    with pytest.raises(ValueError):
        stats.gen_spinsamples(coords, hemi[:-1])
