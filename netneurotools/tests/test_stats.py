# -*- coding: utf-8 -*-
"""
For testing netneurotools.stats functionality
"""

import itertools
import numpy as np
import pytest

from netneurotools import stats


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
    spins, cost = stats.gen_spinsamples(coords, hemi, n_rotate=10, seed=1234)
    assert spins.shape == (len(coords), 10)
    assert len(cost) == 10

    # confirm that `exact` parameter functions as desired
    spin_exact, cost_exact = stats.gen_spinsamples(coords, hemi, n_rotate=10,
                                                   exact=True, seed=1234)
    assert len(spin_exact) == len(coords)
    assert len(spin_exact.T) == len(cost_exact) == 10
    for s in spin_exact.T:
        assert len(np.unique(s)) == len(s)

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

    # only one hemisphere
    # TODO: should this be allowed?
    with pytest.raises(ValueError):
        stats.gen_spinsamples(coords[hemi == 0], hemi[hemi == 0])
