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
    return datasets.fetch_cammoun2012('fsaverage5', data_dir=tmpdir, verbose=0)


@pytest.mark.parametrize('method', [
    'average', 'surface', 'geodesic'
])
@pytest.mark.parametrize('scale, parcels, n_right', [
    ('scale033', 68, 34),
    ('scale060', 114, 57),
    ('scale125', 219, 108),
    ('scale250', 448, 223),
    ('scale500', 1000, 501),
])
def test_find_parcel_centroids(cammoun_surf, scale, parcels, n_right, method):
    lh, rh = cammoun_surf[scale]

    coords, hemi = freesurfer.find_parcel_centroids(lhannot=lh, rhannot=rh,
                                                    method=method,
                                                    version='fsaverage5')
    assert len(coords) == parcels
    assert len(hemi) == parcels
    assert np.sum(hemi) == n_right


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
    assert len(projected) == 20484

    # does reduction return our input data, as expected?
    reduced = freesurfer.vertices_to_parcels(projected, rhannot=rh, lhannot=lh)
    assert np.allclose(data, reduced)

    # can we do this with multi-dimensional data, too?
    data = np.random.rand(parcels, 2)
    projected = freesurfer.parcels_to_vertices(data, rhannot=rh, lhannot=lh)
    assert projected.shape == (20484, 2)
    reduced = freesurfer.vertices_to_parcels(projected, rhannot=rh, lhannot=lh)
    assert np.allclose(data, reduced)

    # what about int arrays as input?
    data = np.random.choice(10, size=parcels)
    projected = freesurfer.parcels_to_vertices(data, rhannot=rh, lhannot=lh)
    reduced = freesurfer.vertices_to_parcels(projected, rhannot=rh, lhannot=lh)
    assert np.allclose(reduced, data)

    # number of parcels != annotation spec
    with pytest.raises(ValueError):
        freesurfer.parcels_to_vertices(np.random.rand(parcels + 1),
                                       rhannot=rh, lhannot=lh)

    # number of vertices != annotation spec
    with pytest.raises(ValueError):
        freesurfer.vertices_to_parcels(np.random.rand(20485),
                                       rhannot=rh, lhannot=lh)
