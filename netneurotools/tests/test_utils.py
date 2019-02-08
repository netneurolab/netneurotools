# -*- coding: utf-8 -*-
"""
For testing netneurotools.utils functionality
"""

import numpy as np
import pytest

from netneurotools import datasets, utils


def test_add_constant():
    # if provided a vector it will return a 2D array
    assert utils.add_constant(np.random.rand(100)).shape == (100, 2)

    # if provided a 2D array it will return the same, extended by 1 column
    out = utils.add_constant(np.random.rand(100, 100))
    assert out.shape == (100, 101) and np.all(out[:, -1] == 1)


def test_add_triu():
    arr = np.arange(9).reshape(3, 3)
    assert np.all(utils.get_triu(arr) == np.array([1, 2, 5]))
    assert np.all(utils.get_triu(arr, k=0) == np.array([0, 1, 2, 4, 5, 8]))


@pytest.mark.parametrize('scale, expected', [
    ('scale033', 83),
    ('scale060', 129),
    ('scale125', 234),
    ('scale250', 463),
    ('scale500', 1015)
])
def test_get_centroids(tmpdir, scale, expected):
    # fetch test dataset
    cammoun = datasets.fetch_cammoun2012('volume', data_dir=tmpdir, verbose=0)

    ijk = utils.get_centroids(cammoun[scale])
    xyz = utils.get_centroids(cammoun[scale], image_space=True)

    # we get expected shape regardless of requested coordinate space
    assert ijk.shape == xyz.shape == (expected, 3)
    # ijk is all positive (i.e., cartesian) coordinates
    assert np.all(ijk > 0)

    # requesting specific labels gives us a subset of the full `ijk`
    lim = utils.get_centroids(cammoun[scale], labels=[1, 2, 3])
    assert np.all(lim == ijk[:3])
