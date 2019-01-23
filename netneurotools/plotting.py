# -*- coding: utf-8 -*-
"""
Functions for making pretty plots and whatnot
"""

import os
from pkg_resources import resource_filename

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns

from .utils import check_fs_subjid


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
        List of labels on {x,y}-axis for each community in `communities`. The
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
                 colorbartitle=None, backgroundcolor=(1, 1, 1),
                 foregroundcolor=(0, 0, 0), **kwargs):

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
    backgroundcolor : tuple of float values with RGB code in [0, 1], optional
        Defines the background color. Default: (1, 1, 1)
    foregroundcolor : tuple of float values with RGB code in [0, 1], optional
        Defines the foreground color (e.g., colorbartitle color).
        Default: (0, 0, 0)
    kwargs : key-value mapping
        Keyword arguments for `mayavi.mlab.triangular_mesh()`

    Returns
    -------
    scene : mayavi.Scene
        Scene object containing plot
    """

    try:
        from mayavi import mlab
    except ImportError:
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

    mlab.figure(bgcolor=backgroundcolor, fgcolor=foregroundcolor,
                figure=lhplot)
    mlab.figure(bgcolor=backgroundcolor, fgcolor=foregroundcolor,
                figure=rhplot)

    return lhplot, rhplot


def plot_fsaverage(data, annot, *, surf='pial', views='lat',
                   tmin=None, tmax=None, center=None, mask=None,
                   colormap='viridis', colorbar=True, alpha=0.8,
                   label_fmt='%.2f', number_of_labels=3,
                   size_per_view=500, subjects_dir=None):
    """
    Plots `data` to fsaverage brain using `annot` as parcellation

    Parameters
    ----------
    data : (N,) array_like
        Data for `N` parcels as defined in `annot`
    annot : str
        Annotation that defines parcellation for `data`. If available in
        fsaverage directory then assumed to be of format '{hemi}.NAME.annot'
        and only NAME should be provided. Otherwise, should provide full path
        to filename but retain the '{hemi}' prefix.
    surf : str, optional
        Surface on which to plot data. Default: 'pial'
    views : str or list, optional
        Which views to plot of brain. Default: 'lat'
    tmin : float, optional
        Minimum value for colorbar. If not provided, a robust estimation will
        be used from values in `data`. Default: None
    tmax : float, optional
        Maximum value for colorbar. If not provided, a robust estimation will
        be used from values in `data`. Default: None
    center : float, optional
        Center of colormap, if desired. Default: None
    mask : (N,) array_like, optional
        Binary array where entries indicate whether values in `data` should be
        masked from plotting (True = mask; False = show). Default: None
    colormap : str, optional
        Which colormap to use for plotting `data`. Default: 'viridis'
    colorbar : bool, optional
        Whether to display the colorbar in the plot. Default: True
    alpha : [0, 1] float, optional
        Transparency of plotted `data`. Default: 0.8
    label_fmt : str, optional
        Format of colorbar labels. Default: '%.2f'
    number_of_labels : int, optional
        Number of labels to display on colorbar. Default: 3
    size_per_view : int, optional
        Size, in pixels, of each frame in the plotted display. Default: 1000
    subjects_dir : str, optional
        Path to FreeSurfer subject directory. If not set, will inherit from
        the environmental variable $SUBJECTS_DIR. Default: None

    Returns
    -------
    brain : surfer.Brain
        Plotted PySurfer brain
    """

    # hold off on imports until
    try:
        from surfer import Brain
    except ImportError:
        raise ImportError('Cannot use plot_to_fsaverage() if pysurfer is not '
                          'installed. Please install pysurfer and try again.')

    subject_id, subjects_dir = check_fs_subjid('fsaverage', subjects_dir)

    # check format of annotation and update, accordingly
    if not annot.endswith('.annot'):
        annot = annot + '.annot'
    if not any([annot.startswith(pref) for pref in ['lh', 'rh', '{hemi}']]):
        annot = '{hemi}.' + annot

    # cast data to float (required for NaNs)
    data = np.asarray(data, dtype='float')

    if mask is not None and len(mask) != len(data):
        raise ValueError('Provided mask must be the same length as data.')

    if tmin is None:
        tmin = np.percentile(data, 2.5)
    if tmax is None:
        tmax = np.percentile(data, 97.5)

    # parcels that should not be included in parcellation
    drop = [b'unknown', b'corpuscallosum']

    # set up brain views
    if not isinstance(views, (np.ndarray, list)):
        views = [views]

    # size of window will depend on # of views provided
    size = (size_per_view * 2, size_per_view * len(views))
    brain = Brain(subject_id='fsaverage', hemi='split', surf=surf,
                  subjects_dir=subjects_dir, background='white',
                  views=views, size=size)

    for hemi in ['lh', 'rh']:
        # loads annotation data for hemisphere, including vertex `labels`!
        if not annot.startswith(os.path.abspath(os.sep)):
            annot_fname = os.path.join(subjects_dir, 'fsaverage', 'label',
                                       annot.format(hemi=hemi))
        else:
            annot_fname = annot.format(hemi=hemi)
        labels, ctab, names = nib.freesurfer.read_annot(annot_fname)

        # get appropriate data, accounting for hemispheric asymmetry
        if hemi == 'lh':
            ldata, rdata = np.split(data, [len(names) - len(drop)])
            if mask is not None:
                lmask, rmask = np.split(mask, [len(names) - len(drop)])
        hemidata = ldata if hemi == 'lh' else rdata

        # our `data` don't include unknown / corpuscallosum, but our `labels`
        # do, so we need to account for that
        # find the label ids that correspond to those and set them to NaN in
        # the `data vector`
        inds = [n for n, f in enumerate(names) if f in drop]
        fulldata = np.insert(hemidata, inds - np.arange(len(inds)), np.nan)
        vtx_data = fulldata[labels]
        not_nan = ~np.isnan(vtx_data)

        # we don't want the NaN vertices (unknown / corpuscallosum) plotted
        # let's drop those and set a threshold so they're hidden
        thresh = vtx_data[not_nan].min() - 1
        vtx_data[np.isnan(vtx_data)] = thresh
        # let's also mask data, if necessary
        if mask is not None:
            maskdata = lmask if hemi == 'lh' else rmask
            maskdata = np.insert(maskdata, inds - np.arange(len(inds)), np.nan)
            vtx_data[maskdata[labels] > 0] = thresh

        # finally, add data to this hemisphere!
        brain.add_data(vtx_data, tmin, tmax, hemi=hemi, mid=center,
                       thresh=thresh + 0.5, alpha=alpha, remove_existing=False,
                       colormap=colormap, colorbar=colorbar, verbose=False)

        # if we have a colorbar, update parameters accordingly
        if colorbar:
            # update label format, as desired
            surf = brain.data_dict[hemi]['surfaces']
            cmap = brain.data_dict[hemi]['colorbars']
            # this updates the format of the colorbar labels
            if label_fmt is not None:
                for n, cm in enumerate(cmap):
                    cm.scalar_bar.label_format = label_fmt
                    surf[n].render()
            # this updates how many labels are shown on the colorbar
            if number_of_labels is not None:
                for n, cm in enumerate(cmap):
                    cm.scalar_bar.number_of_labels = number_of_labels
                    surf[n].render()

    return brain
