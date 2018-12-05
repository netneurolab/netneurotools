# -*- coding: utf-8 -*-
"""
Functions for making pretty plots and whatnot
"""

from pkg_resources import resource_filename

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns

try:
    from mayavi import mlab
    mayavi_avail = True
except ImportError:
    mayavi_avail = False


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


def plot_mod_heatmap(data, communities, *, inds=None, edgecolor='black',
                     ax=None, figsize=(20, 20), xlabels=None, ylabels=None,
                     xlabelrotation=90, ylabelrotation=0, **kwargs):
    """
    Plots `data` as heatmap with borders drawn around `communities`

    Parameters
    ----------
    data : (N, N) array_like
        Correlation matrix
    communities : (N,) array_like
        Community assignments for `data`
    inds : (N,) array_like, optional
        Index array for sorting `data` within `communities`. If None, these
        will be generated from `data`. Default: None
    edgecolor : str, optional
        Color for lines demarcating community boundaries. Default: 'black'
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot the heatmap. If none provided, a new figure and
        axis will be created. Default: None
    figsize : tuple, optional
        Size of figure to create if `ax` is not provided. Default: (20, 20)
    {x,y}labels : list, optional
        List of labels on {x,y}-axis for each community in `communities'. The
        number of labels should match the number of unique communities.
        Default: None
    {x,y}labelrotation : float, optional
        Angle of the rotation of the labels. Available only if `{x,y}labels`
        provided. Default : xlabelrotation: 90, ylabelrotation: 0
    kwargs : key-value mapping
        Keyword arguments for `sns.heatmap()`

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis object containing plot
    """

    # get indices for sorting consensus
    if inds is None:
        inds = sort_communities(data, communities)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    opts = dict(
        ax=ax,
        mask=np.eye(len(data)),
        square=True,
        xticklabels=[],
        yticklabels=[]
    )
    opts.update(**kwargs)

    # plot data re-ordered based on community and node strength
    ax = sns.heatmap(data[np.ix_(inds, inds)], **opts)

    # draw borders around communities
    bounds = _grid_communities(communities)
    bounds[0] += 0.1
    bounds[-1] -= 0.2
    for n, edge in enumerate(np.diff(bounds)):
        ax.add_patch(patches.Rectangle((bounds[n], bounds[n]),
                                       edge, edge, fill=False, linewidth=2,
                                       edgecolor=edgecolor))

    if xlabels is not None or ylabels is not None:
        # find the tick locations
        initloc = _grid_communities(communities)
        tickloc = []
        for loc in range(len(initloc) - 1):
            tickloc.append(np.mean((initloc[loc], initloc[loc + 1])))

        if xlabels is not None:
            # make sure number of labels match the number of ticks
            if len(tickloc) != len(xlabels):
                raise ValueError('Number of labels do not match the number of '
                                 'unique communities.')
            else:
                ax.set_xticks(tickloc)
                ax.set_xticklabels(labels=xlabels, rotation=xlabelrotation)
                ax.tick_params(left=False, bottom=False)
        if ylabels is not None:
            # make sure number of labels match the number of ticks
            if len(tickloc) != len(ylabels):
                raise ValueError('Number of labels do not match the number of '
                                 'unique communities.')
            else:
                ax.set_yticks(tickloc)
                ax.set_yticklabels(labels=ylabels, rotation=ylabelrotation)
                ax.tick_params(left=False, bottom=False)

    return ax


def plot_conte69(data, lhlabel, rhlabel, surf='midthickness',
                 vmin=None, vmax=None, colormap='viridis',
                 colorbar=True, num_labels=4, orientation='horizontal',
                 colorbartitle=None, **kwargs):

    """
    Plots surface `data` on Conte69 Atlas

    Parameters
    ----------
    data : (N,) array_like
        Surface data for N parcels
    lhlabel : str
        Path to .gii file (generic GIFTI file) containing labels to N/2 parcels
        on the left hemisphere
    rhlabel : str
        Path to .gii file (generic GIFTI file) containing labels to N/2 parcels
        on the right hemisphere
    surf : str, optional
        Type of brain surface. Can be 'very_inflated' or 'inflated' or
        'midthickness'. Default: 'midthickness'
    vmin : float, optional
        Minimum value to scale the colormap. If None, the min of the data will
        be used. Default: None
    vmax : float, optional
        Maximum value to scale the colormap. If None, the max of the data will
        be used. Default: None
    colormap : str, optional
        Any colormap from matplotlib. Default: 'viridis'
    colorbar : bool, optional
        Wheter to display a colorbar. Default: True
    num_labels : int, optional
        The number of labels to display on the colorbar.
        Available only if colorbar=True. Default: 4
    orientation : str, optional
        Defines the orientation of colorbar. Can be 'horizontal' or 'vertical'.
        Available only if colorbar=True. Default: 'horizontal'
    colorbartitle : str, optional
        The title of colorbar. Available only if colorbar=True. Default: None
    kwargs : key-value mapping
        Keyword arguments for `mayavi.mlab.triangular_mesh()`

    Returns
    -------
    scene : mayavi.Scene
        Scene object containing plot
    """

    if not mayavi_avail:
        raise ImportError('Cannot use plot_conte69() if mayavi is not '
                          'installed. Please install mayavi and try again.')

    opts = dict()
    opts.update(**kwargs)

    # load surfaces and labels
    lhsurface = nib.load(resource_filename(
        'netneurotools',
        'data/Conte69_Atlas/Conte69.L.%s.32k_fs_LR.surf.gii' % surf))
    rhsurface = nib.load(resource_filename(
        'netneurotools',
        'data/Conte69_Atlas/Conte69.R.%s.32k_fs_LR.surf.gii' % surf))

    lhlabels = nib.load(lhlabel).darrays[0].data
    rhlabels = nib.load(rhlabel).darrays[0].data
    lhvert, lhface = [d.data for d in lhsurface.darrays]
    rhvert, rhface = [d.data for d in rhsurface.darrays]

    # add NaNs for subcortex
    data = np.append(np.nan, data)

    # get lh and rh data
    lhdata = data[lhlabels.astype(int)]
    rhdata = data[rhlabels.astype(int)]

    # plot
    lhplot = mlab.figure()
    rhplot = mlab.figure()
    mlab.triangular_mesh(lhvert[:, 0], lhvert[:, 1], lhvert[:, 2], lhface,
                         figure=lhplot, colormap=colormap,
                         mask=np.isnan(lhdata),
                         scalars=lhdata, vmin=vmin, vmax=vmax, **opts)
    if colorbar is True:
        mlab.colorbar(title=colorbartitle, nb_labels=num_labels,
                      orientation=orientation)
    mlab.triangular_mesh(rhvert[:, 0], rhvert[:, 1], rhvert[:, 2], rhface,
                         figure=rhplot, colormap=colormap,
                         mask=np.isnan(rhdata),
                         scalars=rhdata, vmin=vmin, vmax=vmax, **opts)
    if colorbar is True:
        mlab.colorbar(title=colorbartitle, nb_labels=num_labels,
                      orientation=orientation)
    mlab.view(azimuth=180, elevation=90, distance=450, figure=lhplot)
    mlab.view(azimuth=180, elevation=-90, distance=450, figure=rhplot)

    return lhplot, rhplot
