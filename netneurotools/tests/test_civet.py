# -*- coding: utf-8 -*-
"""
For testing netneurotools.civet functionality
"""

import numpy as np
import pytest

from netneurotools import civet, datasets


@pytest.fixture(scope='module')
def civet_surf(tmp_path_factory):
    tmpdir = str(tmp_path_factory.getbasetemp())
    return datasets.fetch_civet(data_dir=tmpdir, verbose=0)['mid']


def test_read_civet(civet_surf):
    vertices, triangles = civet.read_civet(civet_surf.lh)
    assert len(vertices) == 40962
    assert len(triangles) == 81920
    assert np.all(triangles.max(axis=0) < vertices.shape[0])


def test_civet_to_freesurfer():
    brainmap = np.random.rand(81924)
    out = civet.civet_to_freesurfer(brainmap)
    out2 = civet.civet_to_freesurfer(brainmap, method='linear')
    assert out.shape[0] == out2.shape[0] == 81924

    with pytest.raises(ValueError):
        civet.civet_to_freesurfer(np.random.rand(10))
