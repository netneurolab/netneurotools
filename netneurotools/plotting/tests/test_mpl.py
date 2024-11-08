"""For testing netneurotools.plotting.mpl_plotters functionality."""

import numpy as np
import matplotlib.pyplot as plt
from netneurotools import plotting


def test_grid_communities():
    """Test _grid_communities function."""
    comms = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    # check that comms with / without 0 community label yields same output
    assert np.allclose(plotting._grid_communities(comms), [0, 4, 8, 10])
    assert np.allclose(plotting._grid_communities(comms + 1), [0, 4, 8, 10])


def test_sort_communities():
    """Test sort_communities function."""
    data = np.arange(9).reshape(3, 3)
    comms = np.asarray([0, 0, 2])
    # check that comms with / without 0 community label yields same output
    assert np.allclose(plotting._sort_communities(data, comms), [1, 0, 2])
    assert np.allclose(plotting._sort_communities(data, comms + 1), [1, 0, 2])


def test_plot_mod_heatmap():
    """Test plot_mod_heatmap function."""
    data = np.random.rand(100, 100)
    comms = np.random.choice(4, size=(100,))
    ax = plotting.plot_mod_heatmap(data, comms)
    assert isinstance(ax, plt.Axes)


def test_plot_point_brain():
    """Test plot_point_brain function."""
    data = np.random.rand(100)
    coords = np.random.rand(100, 3)
    out = plotting.plot_point_brain(data, coords)
    assert isinstance(out, plt.Figure)
