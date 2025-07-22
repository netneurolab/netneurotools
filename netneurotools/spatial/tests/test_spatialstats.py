"""For testing netneurotools.spatial.spatial_stats functionality."""

import pytest
import numpy as np

from netneurotools.spatial.spatial_stats import (
    morans_i, local_morans_i,
    gearys_c, local_gearys_c,
    lees_l, local_lees_l
)


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
            3: [0.17645791, 0.03789162, -0.04622557],
            4: [0.40695548, -0.04131271, -0.17433353, 0.52496365],
        },
        "gearys_c": {3: 0.5665884, 4: 0.6756672},
        "local_gearys_c": {
            3: [1.06577412558015, 2.51230240958039, 2.36109222974053],
            4: [
                3.0202751462928, 3.82737772381897,
                3.33112622823278, 2.48027429966568
            ]
        },
        "lees_l": {3: -0.03207622, 4: 0.023249778},
        "local_lees_l": {
            3: [-0.297374681957742, 0.0116826474633146, -0.0293788020084983],
            4: [
                0.439955191359293, -0.0105931628916757,
                -0.253241702688354, 0.351131157157366
            ]
        }
    }

    return annot_1, annot_2, weight, expected_all[which][n]


@pytest.mark.parametrize(
    "generate_test_data",
    [
        pytest.param({"n": n, "which": "morans_i"}, id=f"morans_i-{n}")
        for n in [3, 4]
    ],
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


@pytest.mark.parametrize(
    "generate_test_data",
    [
        pytest.param({"n": n, "which": "gearys_c"}, id=f"gearys_c-{n}")
        for n in [3, 4]
    ],
    indirect=True,
)
def test_gearys_c(generate_test_data):
    """Test Geary's C calculation."""
    annot_1, _, weight, expected = generate_test_data
    assert np.isclose(gearys_c(annot_1, weight, use_numba=False), expected)
    assert np.isclose(gearys_c(annot_1, weight, use_numba=True), expected)


@pytest.mark.parametrize(
    "generate_test_data",
    [
        pytest.param({"n": n, "which": "local_gearys_c"}, id=f"local_gearys_c-{n}")
        for n in [3, 4]
    ],
    indirect=True,
)
def test_local_gearys_c(generate_test_data):
    """Test local Geary's C calculation."""
    annot_1, _, weight, expected = generate_test_data
    assert np.allclose(local_gearys_c(annot_1, weight), expected)


@pytest.mark.parametrize(
    "generate_test_data",
    [
        pytest.param({"n": n, "which": "lees_l"}, id=f"lees_l-{n}")
        for n in [3, 4]
    ],
    indirect=True,
)
def test_lees_l(generate_test_data):
    """Test Lee's L calculation."""
    annot_1, annot_2, weight, expected = generate_test_data
    assert np.isclose(lees_l(annot_1, annot_2, weight, use_numba=False), expected)
    assert np.isclose(lees_l(annot_1, annot_2, weight, use_numba=True), expected)


@pytest.mark.parametrize(
    "generate_test_data",
    [
        pytest.param({"n": n, "which": "local_lees_l"}, id=f"local_lees_l-{n}")
        for n in [3, 4]
    ],
    indirect=True,
)
def test_local_lees_l(generate_test_data):
    """Test local Lee's L calculation."""
    annot_1, annot_2, weight, expected = generate_test_data
    assert np.allclose(local_lees_l(annot_1, annot_2, weight), expected)
