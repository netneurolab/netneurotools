"""For testing netneurotools.metrics.bct functionality."""

import pytest
import numpy as np

from netneurotools import metrics

rs = np.random.RandomState(1234)


def test_communicability_bin():
    """Test communicability_bin function."""
    comm = metrics.communicability_bin(rs.choice([0, 1], size=(100, 100)))
    assert comm.shape == (100, 100)

    with pytest.raises(ValueError):
        metrics.communicability_bin(rs.rand(100, 100))


def test_communicability_wei():
    """Test communicability_wei function."""
    comm = metrics.communicability_wei(rs.rand(100, 100))
    assert comm.shape == (100, 100)
    assert np.allclose(np.diag(comm), 0)
