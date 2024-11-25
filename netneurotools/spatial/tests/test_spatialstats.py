"""For testing netneurotools.spatial.spatial_stats functionality."""

import pytest
import numpy as np

from netneurotools.spatial.spatial_stats import morans_i, local_morans_i


@pytest.fixture(scope="session")
def generate_test_data(request):
    """Generate test data for spatial statistics tests."""
    n = request.param["n"]
    which = request.param["which"]

    rng = np.random.default_rng(1234)
    annot_1 = rng.random(n)
    annot_2 = rng.random(n)
    weight = rng.random((n, n))

    expected_all = {
        "morans_i": {3: 0.04811638, 4: 0.10194804},
        "local_morans_i": {
            3: [0.26468687, 0.05683743, -0.06933836],
            4: [0.54260730, -0.05508362, -0.23244471, 0.69995153],
        },
    }

    return annot_1, annot_2, weight, expected_all[which][n]


@pytest.mark.parametrize(
    "generate_test_data",
    [pytest.param({"n": n, "which": "morans_i"}, id=f"morans_i-{n}") for n in [3, 4]],
    indirect=True,
)
def test_morans_i(generate_test_data):
    """Test Moran's I calculation."""
    annot_1, _, weight, expected = generate_test_data
    assert np.isclose(morans_i(annot_1, weight, use_numba=False), expected)
    assert np.isclose(morans_i(annot_1, weight, use_numba=True), expected)


@pytest.mark.parametrize(
    "generate_test_data",
    [
        pytest.param({"n": n, "which": "local_morans_i"}, id=f"local_morans_i-{n}")
        for n in [3, 4]
    ],
    indirect=True,
)
def test_local_morans_i(generate_test_data):
    """Test local Moran's I calculation."""
    annot_1, _, weight, expected = generate_test_data
    assert np.allclose(local_morans_i(annot_1, weight), expected)
