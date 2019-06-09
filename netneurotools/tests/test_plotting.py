# -*- coding: utf-8 -*-
"""
For testing netneurotools.plotting functionality
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from netneurotools import plotting


@pytest.mark.parametrize('kwargs', [
    dict(vmin=10, vmax=90),
    dict(robust=True),
    dict(xticklabels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
    dict(cmap='RdBu', cbar=False),
    dict(cbar_kws=dict(ticks=[0, 1]))
])
def test_circleplot(kwargs):
    data = np.arange(100).reshape(10, 10)
    fig, ax = plt.subplots(1, 1)
    ax = plotting.circleplot(data, ax=ax, **kwargs)
    assert isinstance(ax, mpl.axes.Axes)
