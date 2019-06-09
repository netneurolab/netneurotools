# -*- coding: utf-8 -*-
"""
For testing netneurotools.plotting functionality
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from netneurotools import plotting


def test_circleplot():
    data = np.arange(100).reshape(10, 10) + 1
    fig, ax = plt.subplots(1, 1)
    ax = plotting.circleplot(data, ax=ax)
    assert isinstance(ax, mpl.axes.Axes)
