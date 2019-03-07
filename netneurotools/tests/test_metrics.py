# -*- coding: utf-8 -*-
"""
For testing netneurotools.metrics functionality
"""

import numpy as np
import pytest

from netneurotools import metrics

rs = np.random.RandomState(1234)


def test_communicability():
    comm = metrics.communicability_bin(rs.choice([0, 1], size=(100, 100)))
    assert comm.shape == (100, 100)

    with pytest.raises(ValueError):
        metrics.communicability_bin(rs.rand(100, 100))


def test_communicability_wei():
    comm = metrics.communicability_wei(rs.rand(100, 100))
    assert comm.shape == (100, 100)
    assert np.allclose(np.diag(comm), 0)
