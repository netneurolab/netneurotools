# -*- coding: utf-8 -*-
"""
For testing netneurotools.plotting functionality
"""

import matplotlib.pyplot as plt
import numpy as np

from netneurotools import datasets, plotting
import pytest


def test_grid_communities():
    comms = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    # check that comms with / without 0 community label yields same output
    assert np.allclose(plotting._grid_communities(comms), [0, 4, 8, 10])
    assert np.allclose(plotting._grid_communities(comms + 1), [0, 4, 8, 10])


def test_sort_communities():
    data = np.arange(9).reshape(3, 3)
    comms = np.asarray([0, 0, 2])
    # check that comms with / without 0 community label yields same output
    assert np.allclose(plotting.sort_communities(data, comms), [1, 0, 2])
    assert np.allclose(plotting.sort_communities(data, comms + 1), [1, 0, 2])


def test_plot_mod_heatmap():
    data = np.random.rand(100, 100)
    comms = np.random.choice(4, size=(100,))
    ax = plotting.plot_mod_heatmap(data, comms)
    assert isinstance(ax, plt.Axes)


@pytest.mark.filterwarnings('ignore')
def test_plot_fsvertex():
    surfer = pytest.importorskip('surfer')

    data = np.random.rand(20484)
    brain = plotting.plot_fsvertex(data, subject_id='fsaverage5',
                                   offscreen=True)
    assert isinstance(brain, surfer.Brain)


@pytest.mark.filterwarnings('ignore')
def test_plot_fsaverage():
    surfer = pytest.importorskip('surfer')

    data = np.random.rand(68)
    lhannot, rhannot = datasets.fetch_cammoun2012('fsaverage5')['scale033']
    brain = plotting.plot_fsaverage(data, lhannot=lhannot, rhannot=rhannot,
                                    subject_id='fsaverage5', offscreen=True)
    assert isinstance(brain, surfer.Brain)


def test_plot_point_brain():
    data = np.random.rand(100)
    coords = np.random.rand(100, 3)
    out = plotting.plot_point_brain(data, coords)
    assert isinstance(out, plt.Figure)
