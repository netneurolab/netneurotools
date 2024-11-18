"""For testing netneurotools.stats.regression functionality."""

import numpy as np
from netneurotools import stats


def test_add_constant():
    """Test adding a constant to a 1D or 2D array."""
    # if provided a vector it will return a 2D array
    assert stats._add_constant(np.random.rand(100)).shape == (100, 2)

    # if provided a 2D array it will return the same, extended by 1 column
    out = stats._add_constant(np.random.rand(100, 100))
    assert out.shape == (100, 101) and np.all(out[:, -1] == 1)
