"""Functional tests for the transforms module."""

import numpy as np
import pytest
from netneurotools.interface.surf_parc import (
    parcels_to_vertices,  # noqa: F401
    vertices_to_parcels,
)
from netneurotools.datasets import fetch_schaefer2018, fetch_cammoun2012

_parc_to_nvert = {
    "annot": 10242,
    "dlabel.nii": 32492,
    "gii": 32492,
}
_parc_to_nparc = {
    "annot": 50,
    "dlabel.nii": 50,
    "gii": 34,
}


test_params, test_ids = zip(
    *[
        (
            (
                {"file_type": file_type, "hemi_type": hemi_type},
                {
                    "file_type": file_type,
                    "hemi_type": hemi_type,
                    "data_width": data_width,
                },
            ),
            f"{file_type}-{hemi_type}-{data_width}d",
        )
        for file_type in ["annot", "dlabel.nii", "gii"]
        for hemi_type in ["both", "L", "R"]
        for data_width in [0, 1, 2]
    ]
)


@pytest.fixture(scope="session")
def prepare_parc_files(request, tmp_path_factory):
    """Prepare parcellation files for parcels_to_vertices tests."""
    file_type = request.param["file_type"]
    hemi_type = request.param["hemi_type"]
    data_dir = tmp_path_factory.mktemp("data")

    if file_type == "annot":
        parc = fetch_schaefer2018(version="fsaverage5", data_dir=data_dir, verbose=0)[
            "100Parcels7Networks"
        ]
    elif file_type == "dlabel.nii":
        parc = fetch_schaefer2018(version="fslr32k", data_dir=data_dir, verbose=0)[
            "100Parcels7Networks"
        ]
    elif file_type == "gii":
        parc = fetch_cammoun2012(version="fslr32k", data_dir=data_dir, verbose=0)[
            "scale033"
        ]
    else:
        raise ValueError(f"Invalid file type: {file_type}")

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

    return parc_file


def _generate_random_data(which, file_type, hemi_type, data_width):
    if which == "vertices_to_parcels":
        n = _parc_to_nvert[file_type]
    elif which == "parcels_to_vertices":
        n = _parc_to_nparc[file_type]
    else:
        raise ValueError(f"Invalid function: {which}")

    rng = np.random.default_rng(sum(map(ord, file_type + hemi_type)))
    data = (rng.random(n), rng.random(n))

    if data_width == 0:
        pass
    elif data_width == 1:
        data = (data[0][:, np.newaxis], data[1][:, np.newaxis])
    elif data_width == 2:
        data = (np.c_[data[0], data[0]], np.c_[data[1], data[1]])
    else:
        raise ValueError(f"Invalid data dimension: {data_width}")

    if hemi_type == "both":
        data = data
    elif hemi_type == "L":
        data = data[0]
    elif hemi_type == "R":
        data = data[1]
    else:
        raise ValueError(f"Invalid hemisphere type: {hemi_type}")

    return data


@pytest.fixture(scope="session")
def generate_vertices_to_parcels_data(request):
    """Generate data for vertices_to_parcels tests."""
    file_type = request.param["file_type"]
    hemi_type = request.param["hemi_type"]
    data_width = request.param["data_width"]

    vert_data = _generate_random_data(
        "vertices_to_parcels", file_type, hemi_type, data_width
    )

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

    return vert_data, expected


@pytest.mark.parametrize(
    "prepare_parc_files, generate_vertices_to_parcels_data",
    test_params,
    ids=test_ids,
    indirect=True,
)
def test_vertices_to_parcels(
    prepare_parc_files, generate_vertices_to_parcels_data, request
):
    """Test vertices_to_parcels function."""
    # curr_id = request.node.callspec.id
    curr_params = request.node.callspec.params["generate_vertices_to_parcels_data"]
    # file_type = curr_params["file_type"]
    hemi_type = curr_params["hemi_type"]
    data_width = curr_params["data_width"]

    parc_file = prepare_parc_files
    vert_data, expected = generate_vertices_to_parcels_data

    reduced, keys, labels = vertices_to_parcels(
        vert_data, parc_file, hemi=hemi_type, background=0
    )
    if hemi_type == "both":
        # test output shape and type
        assert len(reduced) == 2
        assert all([isinstance(k, np.ndarray) for k in reduced])
        assert len(keys) == 2
        assert all([isinstance(k, tuple) for k in keys])
        assert len(labels) == 2
        assert all([isinstance(k, tuple) for k in labels])
        # test output data
        if data_width == 0:
            assert all([r.ndim == 1 for r in reduced])
        else:
            assert all([r.ndim == 2 for r in reduced])
            assert all([r.shape[1] == data_width for r in reduced])
        assert sum([r.shape[0] for r in reduced]) == expected[0]
        assert np.allclose(np.mean(np.concatenate(reduced), axis=0), expected[1])
    else:
        # test output shape and type
        assert isinstance(reduced, np.ndarray)
        assert isinstance(keys, tuple)
        assert isinstance(labels, tuple)
        # test output data
        if data_width == 0:
            assert reduced.ndim == 1
        else:
            assert reduced.ndim == 2
            assert reduced.shape[1] == data_width

        assert reduced.shape[0] == expected[0]
        assert np.allclose(np.mean(reduced, axis=0), expected[1])


@pytest.fixture(scope="session")
def generate_parcels_to_vertices_data(request):
    """Generate data for parcels_to_vertices tests."""
    file_type = request.param["file_type"]
    hemi_type = request.param["hemi_type"]
    data_width = request.param["data_width"]

    parcels_data = _generate_random_data(
        "parcels_to_vertices", file_type, hemi_type, data_width
    )

    expected_all = {
        "annot": {
            "both": (20484, 0.48047245),
            "L": (10242, 0.5117257),
            "R": (10242, 0.49651137),
        },
        "dlabel.nii": {
            "both": (64984, 0.52227944),
            "L": (32492, 0.5166443),
            "R": (32492, 0.46440256),
        },
        "gii": {
            "both": (64984, 0.5227571),
            "L": (32492, 0.54153746),
            "R": (32492, 0.5344661),
        },
    }
    # generate expected output
    expected = expected_all[file_type][hemi_type]

    return parcels_data, expected


@pytest.mark.parametrize(
    "prepare_parc_files, generate_parcels_to_vertices_data",
    test_params,
    ids=test_ids,
    indirect=True,
)
def test_parcels_to_vertices(
    prepare_parc_files, generate_parcels_to_vertices_data, request
):
    """Test parcels_to_vertices function."""
    curr_params = request.node.callspec.params["generate_parcels_to_vertices_data"]
    hemi_type = curr_params["hemi_type"]
    data_width = curr_params["data_width"]

    parc_file = prepare_parc_files
    parc_data, expected = generate_parcels_to_vertices_data

    projected, keys, labels = parcels_to_vertices(
        parc_data, parc_file, hemi=hemi_type, fill=np.nan
    )
    if hemi_type == "both":
        # test output shape and type
        assert len(projected) == 2
        assert all([isinstance(k, np.ndarray) for k in projected])
        assert len(keys) == 2
        assert all([isinstance(k, tuple) for k in keys])
        assert len(labels) == 2
        assert all([isinstance(k, tuple) for k in labels])
        # test output data
        if data_width == 0:
            assert all([r.ndim == 1 for r in projected])
        else:
            assert all([r.ndim == 2 for r in projected])
            assert all([r.shape[1] == data_width for r in projected])
        assert sum([r.shape[0] for r in projected]) == expected[0]
        assert np.allclose(np.nanmean(np.concatenate(projected), axis=0), expected[1])
    else:
        # test output shape and type
        assert isinstance(projected, np.ndarray)
        assert isinstance(keys, tuple)
        assert isinstance(labels, tuple)
        # test output data
        if data_width == 0:
            assert projected.ndim == 1
        else:
            assert projected.ndim == 2
            assert projected.shape[1] == data_width

        assert projected.shape[0] == expected[0]
        assert np.allclose(np.nanmean(projected, axis=0), expected[1])
