"""Functional tests for the transforms module."""

import numpy as np
import pytest
from netneurotools.interface.transforms import (
    parcels_to_vertices,  # noqa: F401
    vertices_to_parcels,
)
from netneurotools.datasets import fetch_schaefer2018, fetch_cammoun2012


@pytest.fixture(scope="session")
def generate_vertices_to_parcels_data(request, tmp_path_factory):
    """Generate data for vertices_to_parcels tests."""
    file_type = request.param["file_type"]
    hemi_type = request.param["hemi_type"]
    data_width = request.param["data_width"]

    data_dir = tmp_path_factory.mktemp("data")

    # load parcellations
    if file_type == "annot":
        parc = fetch_schaefer2018(version="fsaverage5", data_dir=data_dir, verbose=0)[
            "100Parcels7Networks"
        ]
        n_vert = 10242
    elif file_type == "dlabel.nii":
        parc = fetch_schaefer2018(version="fslr32k", data_dir=data_dir, verbose=0)[
            "100Parcels7Networks"
        ]
        n_vert = 32492
    elif file_type == "gii":
        parc = fetch_cammoun2012(version="fslr32k", data_dir=data_dir, verbose=0)[
            "scale033"
        ]
        n_vert = 32492
    else:
        raise ValueError(f"Invalid file type: {file_type}")

    # handle hemispheres and write to parc_file
    if file_type == "dlabel.nii":
        parc_file = parc
    else:
        if hemi_type == "both":
            parc_file = (parc.L, parc.R)
        elif hemi_type == "L":
            parc_file = parc.L
        elif hemi_type == "R":
            parc_file = parc.R
        else:
            raise ValueError(f"Invalid hemisphere type: {hemi_type}")

    # generate random vertex data
    rng = np.random.default_rng(sum(map(ord, file_type + hemi_type)))
    vert = (rng.random(n_vert), rng.random(n_vert))
    if data_width == 0:
        pass
    elif data_width == 1:
        vert = (vert[0][:, np.newaxis], vert[1][:, np.newaxis])
    elif data_width == 2:
        vert = (np.c_[vert[0], vert[0]], np.c_[vert[1], vert[1]])
    else:
        raise ValueError(f"Invalid data dimension: {data_width}")

    if hemi_type == "both":
        vert_data = vert
    elif hemi_type == "L":
        vert_data = vert[0]
    elif hemi_type == "R":
        vert_data = vert[1]
    else:
        pass

    expected_all = {
        "annot": {
            "both": (100, 0.4989233),
            "L": (50, 0.49557236),
            "R": (50, 0.49596187),
        },
        "dlabel.nii": {
            "both": (100, 0.497879),
            "L": (50, 0.49634945),
            "R": (50, 0.5019505),
        },
        "gii": {
            "both": (68, 0.4966058),
            "L": (34, 0.49639294),
            "R": (34, 0.5032322),
        },
    }
    # generate expected output
    expected = expected_all[file_type][hemi_type]

    return parc_file, vert_data, expected


@pytest.mark.parametrize(
    "generate_vertices_to_parcels_data",
    [
        pytest.param(
            {"file_type": file_type, "hemi_type": hemi_type, "data_width": data_width},
            id=f"{file_type}-{hemi_type}-{data_width}d",
        )
        for file_type in ["annot", "dlabel.nii", "gii"]
        for hemi_type in ["both", "L", "R"]
        for data_width in [0, 1, 2]
    ],
    indirect=True,
)
def test_vertices_to_parcels(generate_vertices_to_parcels_data, request):
    """Test vertices_to_parcels function."""
    # curr_id = request.node.callspec.id
    curr_params = request.node.callspec.params["generate_vertices_to_parcels_data"]
    # file_type = curr_params["file_type"]
    hemi_type = curr_params["hemi_type"]
    data_width = curr_params["data_width"]

    parc_file, vert_data, expected = generate_vertices_to_parcels_data
    reduced, keys, labels = vertices_to_parcels(
        vert_data, parc_file, hemi=hemi_type, background=0
    )
    if hemi_type == "both":
        assert len(reduced) == 2
        assert all([isinstance(k, np.ndarray) for k in reduced])
        assert len(keys) == 2
        assert all([isinstance(k, tuple) for k in keys])
        assert len(labels) == 2
        assert all([isinstance(k, tuple) for k in labels])
        #
        if data_width == 0:
            assert all([r.ndim == 1 for r in reduced])
        else:
            assert all([r.ndim == 2 for r in reduced])
            assert all([r.shape[1] == data_width for r in reduced])
        assert sum([r.shape[0] for r in reduced]) == expected[0]
        assert np.allclose(np.mean(np.concatenate(reduced), axis=0), expected[1])
    else:
        assert isinstance(reduced, np.ndarray)
        assert isinstance(keys, tuple)
        assert isinstance(labels, tuple)
        #
        if data_width == 0:
            assert reduced.ndim == 1
        else:
            assert reduced.ndim == 2
            assert reduced.shape[1] == data_width

        assert reduced.shape[0] == expected[0]
        assert np.allclose(np.mean(reduced, axis=0), expected[1])


def test_parcels_to_vertices(generate_parcels_to_vertices_data, request):
    """Test parcels_to_vertices function."""
    pass
