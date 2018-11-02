# -*- coding: utf-8 -*-
"""
Functions for making pretty plots and whatnot
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _grid_communities(communities):
    """
    Generates boundaries of `communities`

    Parameters
    ----------
    communities : array_like
        Community assignment vector

    Returns
    -------
    bounds : list
        Boundaries of communities
    """

    comm = communities[np.argsort(communities)]
    bounds = []
    for i in range(1, np.max(comm) + 1):
        ind = np.where(comm == i)
        if len(ind) > 0:
            bounds.append(np.min(ind))

    bounds.append(len(communities))

    return bounds


def sort_communities(consensus, communities):
    """
    Sorts ``communities`` in ``consensus`` according to strength

    Parameters
    ----------
    consensus : array_like
        Correlation matrix
    communities : array_like
        Community assignments for ``consensus``

    Returns
    -------
    inds : np.ndarray
        Index array for sorting ``consensus``
    """

    if 0 in communities:
        communities += 1

    bounds = _grid_communities(communities)
    inds = np.argsort(communities)

    for n, f in enumerate(bounds[:-1]):
        i = inds[f:bounds[n + 1]]
        cco = i[consensus[np.ix_(i, i)].mean(axis=1).argsort()[::-1]]
        inds[f:bounds[n + 1]] = cco

    return inds


def plot_mod_heatmap(consensus, communities, *, inds=None, edgecolor='black',
                     ax=None, figsize=(20, 20), **kwargs):
    """
    Plots `consensus` as heatmap with borders drawn around `communities`

    Parameters
    ----------
    consensus : (N, N) array_like
        Correlation matrix
    communities : (N,) array_like
        Community assignments for correlation matrix
    inds : (N,) array_like, optional
        Index array for sorting `consensus` within `communities`. If None,
        these will be generated from `consensus`. Default: None
    edgecolor : str, optional
        Color for lines demarcating community boundaries. Default: 'black'
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot the heatmap. If none provided, a new figure and
        axis will be created. Default: None
    figsize : tuple, optional
        Size of figure to create if `ax` is not provided. Default: (20, 20)
    kwargs : key-value mapping
        Keyword arguments for `sns.heatmap()`

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis object containing plot
    """

    # get indices for sorting consensus
    if inds is None:
        inds = sort_communities(consensus, communities)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax = sns.heatmap(consensus[np.ix_(inds, inds)], ax=ax,
                     mask=np.eye(len(consensus)), square=True,
                     xticklabels=[], yticklabels=[],
                     **kwargs)

    # draw borders around communities
    bounds = _grid_communities(communities)
    bounds[0] += 0.1
    bounds[-1] -= 0.2
    for n, edge in enumerate(np.diff(bounds)):
        ax.add_patch(patches.Rectangle((bounds[n], bounds[n]),
                                       edge, edge,
                                       fill=False, linewidth=2,
                                       edgecolor=edgecolor))

    return ax
