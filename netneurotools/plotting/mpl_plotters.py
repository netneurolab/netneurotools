"""Functions for matplotlib-based plotting."""

from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _grid_communities(communities):
    """
    Generate boundaries of `communities`.

    Parameters
    ----------
    communities : array_like
        Community assignment vector

    Returns
    -------
    bounds : list
        Boundaries of communities
    """
    communities = np.asarray(communities)
    if 0 in communities:
        communities = communities + 1

    comm = communities[np.argsort(communities)]
    bounds = []
    for i in np.unique(comm):
        ind = np.where(comm == i)
        if len(ind) > 0:
            bounds.append(np.min(ind))

    bounds.append(len(communities))

    return bounds


def _sort_communities(consensus, communities):
    """
    Sort `communities` in `consensus` according to strength.

    Parameters
    ----------
    consensus : array_like
        Correlation matrix
    communities : array_like
        Community assignments for `consensus`

    Returns
    -------
    inds : np.ndarray
        Index array for sorting `consensus`
    """
    communities = np.asarray(communities)
    if 0 in communities:
        communities = communities + 1

    bounds = _grid_communities(communities)
    inds = np.argsort(communities)

    for n, f in enumerate(bounds[:-1]):
        i = inds[f:bounds[n + 1]]
        cco = i[consensus[np.ix_(i, i)].mean(axis=1).argsort()[::-1]]
        inds[f:bounds[n + 1]] = cco

    return inds


def plot_mod_heatmap(data, communities, *, inds=None, edgecolor='black',
                     ax=None, figsize=(6.4, 4.8), xlabels=None, ylabels=None,
                     xlabelrotation=90, ylabelrotation=0, cbar=True,
                     square=True, xticklabels=None, yticklabels=None,
                     mask_diagonal=True, **kwargs):
    """
    Plot `data` as heatmap with borders drawn around `communities`.

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
        List of labels on {x,y}-axis for each community in `communities`. The
        number of labels should match the number of unique communities.
        Default: None
    {x,y}labelrotation : float, optional
        Angle of the rotation of the labels. Available only if `{x,y}labels`
        provided. Default : xlabelrotation: 90, ylabelrotation: 0
    square : bool, optional
        Setting the matrix with equal aspect. Default: True
    {x,y}ticklabels : list, optional
        Incompatible with `{x,y}labels`. List of labels for each entry (not
        community) in `data`. Default: None
    cbar : bool, optional
        Whether to plot colorbar. Default: True
    mask_diagonal : bool, optional
        Whether to mask the diagonal in the plotted heatmap. Default: True
    kwargs : key-value mapping
        Keyword arguments for `plt.pcolormesh()`

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis object containing plot
    """
    for t, label in zip([xticklabels, yticklabels], [xlabels, ylabels]):
        if t is not None and label is not None:
            raise ValueError('Cannot set both {x,y}labels and {x,y}ticklabels')

    # get indices for sorting consensus
    if inds is None:
        inds = _sort_communities(data, communities)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    # plot data re-ordered based on community and node strength
    if mask_diagonal:
        plot_data = np.ma.masked_where(np.eye(len(data)),
                                       data[np.ix_(inds, inds)])
    else:
        plot_data = data[np.ix_(inds, inds)]

    coll = ax.pcolormesh(plot_data, edgecolor='none', **kwargs)
    ax.set(xlim=(0, plot_data.shape[1]), ylim=(0, plot_data.shape[0]))

    # set equal aspect
    if square:
        ax.set_aspect('equal')

    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)

    # invert the y-axis so it looks "as expected"
    ax.invert_yaxis()

    # plot the colorbar
    if cbar:
        cb = ax.figure.colorbar(coll)
        if kwargs.get('rasterized', False):
            cb.solids.set_rasterized(True)

    # draw borders around communities
    bounds = _grid_communities(communities)
    bounds[0] += 0.2
    bounds[-1] -= 0.2
    for n, edge in enumerate(np.diff(bounds)):
        ax.add_patch(mpatches.Rectangle((bounds[n], bounds[n]),
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

    if xticklabels is not None:
        labels_ind = [xticklabels[i] for i in inds]
        ax.set_xticks(np.arange(len(labels_ind)) + 0.5)
        ax.set_xticklabels(labels_ind, rotation=90)
    if yticklabels is not None:
        labels_ind = [yticklabels[i] for i in inds]
        ax.set_yticks(np.arange(len(labels_ind)) + 0.5)
        ax.set_yticklabels(labels_ind)

    return ax


def plot_point_brain(data, coords, views=None, views_orientation='vertical',
                     views_size=(4, 2.4), cbar=False, robust=True, size=50,
                     **kwargs):
    """
    Plot `data` as a cloud of points in 3D space based on specified `coords`.

    Parameters
    ----------
    data : (N,) array_like
        Data for an `N` node parcellation; determines color of points
    coords : (N, 3) array_like
        x, y, z coordinates for `N` node parcellation
    views : list, optional
        List specifying which views to use. Can be any of {'sagittal', 'sag',
        'coronal', 'cor', 'axial', 'ax'}. If not specified will use 'sagittal'
        and 'axial'. Default: None
    views_orientation: str, optional
        Orientation of the views. Can be either 'vertical' or 'horizontal'.
        Default: 'vertical'.
    views_size : tuple, optional
        Figure size of each view. Default: (4, 2.4)
    cbar : bool, optional
        Whether to also show colorbar. Default: False
    robust : bool, optional
        Whether to use robust calculation of `vmin` and `vmax` for color scale.
    size : int, optional
        Size of points on plot. Default: 50
    **kwargs
        Key-value pairs passed to `matplotlib.axes.Axis.scatter`

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
    """
    _views = dict(sagittal=(0, 180), sag=(0, 180),
                  axial=(90, 180), ax=(90, 180),
                  coronal=(0, 90), cor=(0, 90))

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    if views is None:
        views = [_views[f] for f in ['sagittal', 'axial']]
    else:
        if not isinstance(views, Iterable) or isinstance(views, str):
            views = [views]
        views = [_views[f] for f in views]

    if views_orientation == 'vertical':
        ncols, nrows = 1, len(views)
    elif views_orientation == 'horizontal':
        ncols, nrows = len(views), 1
    figsize = (ncols * views_size[0], nrows * views_size[1])

    # create figure and axes (3d projections)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize=figsize,
                             subplot_kw=dict(projection='3d'))

    opts = dict(linewidth=0.5, edgecolor='gray', cmap='viridis')
    if robust:
        vmin, vmax = np.percentile(data, [2.5, 97.5])
        opts.update(dict(vmin=vmin, vmax=vmax))
    opts.update(kwargs)

    # iterate through saggital/axial views and plot, rotating as needed
    for n, view in enumerate(views):
        # if only one view then axes is not a list!
        ax = axes[n] if len(views) > 1 else axes
        # make the actual scatterplot and update the view / aspect ratios
        col = ax.scatter(x, y, z, c=data, s=size, **opts)
        ax.view_init(*view)
        ax.axis('off')
        scaling = np.array([ax.get_xlim(),
                            ax.get_ylim(),
                            ax.get_zlim()])
        ax.set_box_aspect(tuple(scaling[:, 1] - scaling[:, 0]))

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)

    # add colorbar to axes
    if cbar:
        cbar = fig.colorbar(col, ax=axes.flatten(),
                            drawedges=False, shrink=0.7)
        cbar.outline.set_linewidth(0)

    return fig
