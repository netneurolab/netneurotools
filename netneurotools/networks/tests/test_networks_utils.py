"""For testing netneurotools.networks.networks_utils functionality."""

import numpy as np

from netneurotools import networks


def test_get_triu():
    """Test that get_triu returns correct values."""
    arr = np.arange(9).reshape(3, 3)
    assert np.all(networks.get_triu(arr) == np.array([1, 2, 5]))
    assert np.all(networks.get_triu(arr, k=0) == np.array([0, 1, 2, 4, 5, 8]))
