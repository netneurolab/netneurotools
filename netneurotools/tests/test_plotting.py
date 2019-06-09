# -*- coding: utf-8 -*-
"""
For testing netneurotools.plotting functionality
"""

import matplotlib.pyplot as plt
import numpy as np

from netneurotools import plotting

rs = np.random.RandomState(1234)


def test_circleplot():
    data = rs.random.rand(10, 10)
    fig, ax = plt.subplots(1, 1)
    plotting.circleplot(data, ax=ax)
