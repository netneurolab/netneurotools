# -*- coding: utf-8 -*-
"""
For testing netneurotools.freesurfer functionality
"""

import numpy as np
import pytest

from netneurotools import datasets, freesurfer


@pytest.fixture(scope='module')
def cammoun_surf(tmp_path_factory):
    tmpdir = str(tmp_path_factory.getbasetemp())
    return datasets.fetch_cammoun2012('fsaverage', data_dir=tmpdir, verbose=0)


@pytest.mark.parametrize('scale, parcels', [
    ('scale033', 68),
    ('scale060', 114),
    ('scale125', 219),
    ('scale250', 448),
    ('scale500', 1000),
])
def test_project_reduce_vertices(cammoun_surf, scale, parcels):
    # these functions are partners and should be tested in concert.
    # we can test all the normal functionality and also ensure that "round
    # trips" work as expected

    # generate "parcellated" data
    data = np.random.rand(parcels)
    lh, rh = cammoun_surf[scale]

    # do we get the expected number of vertices in our projection?
    projected = freesurfer.parcels_to_vertices(data, rhannot=rh, lhannot=lh)
    assert len(projected) == 327684

    # does reduction return our input data, as expected?
    reduced = freesurfer.vertices_to_parcels(projected, rhannot=rh, lhannot=lh)
    assert np.allclose(data, reduced)

    # can we do this with multi-dimensional data, too?
    data = np.random.rand(parcels, 2)
    projected = freesurfer.parcels_to_vertices(data, rhannot=rh, lhannot=lh)
    assert projected.shape == (327684, 2)
    reduced = freesurfer.vertices_to_parcels(projected, rhannot=rh, lhannot=lh)
    assert np.allclose(data, reduced)

    # number of parcels != annotation spec
    with pytest.raises(ValueError):
        freesurfer.parcels_to_vertices(np.random.rand(parcels + 1),
                                       rhannot=rh, lhannot=lh)

    # number of vertices != annotation spec
    with pytest.raises(ValueError):
        freesurfer.vertices_to_parcels(np.random.rand(327685),
                                       rhannot=rh, lhannot=lh)
