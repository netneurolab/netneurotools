# -*- coding: utf-8 -*-
"""
Functions for making pretty plots and whatnot
"""

import os
from typing import Iterable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import nibabel as nib
import numpy as np

from .freesurfer import FSIGNORE, _decode_list


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


def sort_communities(consensus, communities):
    """
    Sorts `communities` in `consensus` according to strength

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

    for t, l in zip([xticklabels, yticklabels], [xlabels, ylabels]):
        if t is not None and l is not None:
            raise ValueError('Cannot set both {x,y}labels and {x,y}ticklabels')

    # get indices for sorting consensus
    if inds is None:
        inds = sort_communities(data, communities)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

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

    if xticklabels is not None:
        labels_ind = [xticklabels[i] for i in inds]
        ax.set_xticks(np.arange(len(labels_ind)) + 0.5)
        ax.set_xticklabels(labels_ind, rotation=90)
    if yticklabels is not None:
        labels_ind = [yticklabels[i] for i in inds]
        ax.set_yticks(np.arange(len(labels_ind)) + 0.5)
        ax.set_yticklabels(labels_ind)

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
    surf : {'midthickness', 'inflated', 'vinflated'}, optional
        Type of brain surface. Default: 'midthickness'
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

    return plot_fslr(data, lhlabel, rhlabel, surf_atlas='conte69',
                     surf_type=surf, vmin=vmin, vmax=vmax, colormap=colormap,
                     colorbar=colorbar, num_labels=num_labels,
                     orientation=orientation, colorbartitle=colorbartitle,
                     backgroundcolor=backgroundcolor,
                     foregroundcolor=foregroundcolor, **kwargs)


def plot_fslr(data, lhlabel, rhlabel, surf_atlas='conte69',
              surf_type='midthickness', vmin=None, vmax=None,
              colormap='viridis', colorbar=True, num_labels=4,
              orientation='horizontal', colorbartitle=None,
              backgroundcolor=(1, 1, 1), foregroundcolor=(0, 0, 0),
              **kwargs):

    """
    Plots surface `data` on a given fsLR32k atlas

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
    surf_atlas: {'conte69', 'yerkes19'}, optional
        Surface atlas on which to plot 'data'. Default: 'conte69'
    surf_type : {'midthickness', 'inflated', 'vinflated'}, optional
        Type of brain surface. Default: 'midthickness'
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

    from .datasets import fetch_conte69, fetch_yerkes19
    try:
        from mayavi import mlab
    except ImportError:
        raise ImportError('Cannot use plot_fslr() if mayavi is not '
                          'installed. Please install mayavi and try again.')

    opts = dict()
    opts.update(**kwargs)

    try:
        if surf_atlas == 'conte69':
            surface = fetch_conte69()[surf_type]
        elif surf_atlas == 'yerkes19':
            surface = fetch_yerkes19()[surf_type]

    except KeyError:
        raise ValueError('Provided surf "{}" is not valid. Must be one of '
                         '[\'midthickness\', \'inflated\', \'vinflated\']'
                         .format(surf_type))

    lhsurface, rhsurface = [nib.load(s) for s in surface]

    lhlabels = nib.load(lhlabel).darrays[0].data
    rhlabels = nib.load(rhlabel).darrays[0].data
    lhvert, lhface = [d.data for d in lhsurface.darrays]
    rhvert, rhface = [d.data for d in rhsurface.darrays]

    # add NaNs for medial wall
    data = np.append(np.nan, data)

    # get lh and rh data
    lhdata = np.squeeze(data[lhlabels.astype(int)])
    rhdata = np.squeeze(data[rhlabels.astype(int)])

    # plot
    lhplot = mlab.figure()
    rhplot = mlab.figure()
    lhmesh = mlab.triangular_mesh(lhvert[:, 0], lhvert[:, 1], lhvert[:, 2],
                                  lhface, figure=lhplot, colormap=colormap,
                                  mask=np.isnan(lhdata), scalars=lhdata,
                                  vmin=vmin, vmax=vmax, **opts)
    lhmesh.module_manager.scalar_lut_manager.lut.nan_color = [0.863, 0.863,
                                                              0.863, 1]
    lhmesh.update_pipeline()
    if colorbar is True:
        mlab.colorbar(title=colorbartitle, nb_labels=num_labels,
                      orientation=orientation)
    rhmesh = mlab.triangular_mesh(rhvert[:, 0], rhvert[:, 1], rhvert[:, 2],
                                  rhface, figure=rhplot, colormap=colormap,
                                  mask=np.isnan(rhdata), scalars=rhdata,
                                  vmin=vmin, vmax=vmax, **opts)
    rhmesh.module_manager.scalar_lut_manager.lut.nan_color = [0.863, 0.863,
                                                              0.863, 1]
    rhmesh.update_pipeline()
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


def _get_fs_subjid(subject_id, subjects_dir=None):
    """
    Gets fsaverage version `subject_id`, fetching if required

    Parameters
    ----------
    subject_id : str
        FreeSurfer subject ID
    subjects_dir : str, optional
        Path to FreeSurfer subject directory. If not set, will inherit from
        the environmental variable $SUBJECTS_DIR. Default: None

    Returns
    -------
    subject_id : str
        FreeSurfer subject ID
    subjects_dir : str
        Path to subject directory with `subject_id`
    """

    from netneurotools.utils import check_fs_subjid

    # check for FreeSurfer install w/fsaverage; otherwise, fetch required
    try:
        subject_id, subjects_dir = check_fs_subjid(subject_id, subjects_dir)
    except FileNotFoundError:
        if 'fsaverage' not in subject_id:
            raise ValueError('Provided subject {} does not exist in provided '
                             'subjects_dir {}'
                             .format(subject_id, subjects_dir))
        from netneurotools.datasets import fetch_fsaverage
        from netneurotools.datasets.utils import _get_data_dir
        fetch_fsaverage(subject_id)
        subjects_dir = os.path.join(_get_data_dir(), 'tpl-fsaverage')
        subject_id, subjects_dir = check_fs_subjid(subject_id, subjects_dir)

    return subject_id, subjects_dir


def plot_fsaverage(data, *, lhannot, rhannot, order='lr', mask=None,
                   noplot=None, subject_id='fsaverage', subjects_dir=None,
                   vmin=None, vmax=None, **kwargs):
    """
    Plots `data` to fsaverage brain using `annot` as parcellation

    Parameters
    ----------
    data : (N,) array_like
        Data for `N` parcels as defined in `annot`
    lhannot : str
        Filepath to .annot file containing labels to parcels on the left
        hemisphere. If a full path is not provided the file is assumed to
        exist inside the `subjects_dir`/`subject`/label directory.
    rhannot : str
        Filepath to .annot file containing labels to parcels on the right
        hemisphere. If a full path is not provided the file is assumed to
        exist inside the `subjects_dir`/`subject`/label directory.
    order : str, optional
        Order of the hemispheres in the data vector (either 'LR' or 'RL').
        Default: 'LR'
    mask : (N,) array_like, optional
        Binary array where entries indicate whether values in `data` should be
        masked from plotting (True = mask; False = show). Default: None
    noplot : list, optional
        List of names in `lhannot` and `rhannot` to not plot. It is assumed
        these are NOT present in `data`. By default 'unknown' and
        'corpuscallosum' will never be plotted if they are present in the
        provided annotation files. Default: None
    subject_id : str, optional
        Subject ID to use; must be present in `subjects_dir`. Default:
        'fsaverage'
    subjects_dir : str, optional
        Path to FreeSurfer subject directory. If not set, will inherit from
        the environmental variable $SUBJECTS_DIR. Default: None
    vmin : float, optional
        Minimum value for colorbar. If not provided, a robust estimation will
        be used from values in `data`. Default: None
    vmax : float, optional
        Maximum value for colorbar. If not provided, a robust estimation will
        be used from values in `data`. Default: None
    kwargs : key-value pairs
        Provided directly to :func:`~.plot_fsvertex` without modification.

    Returns
    -------
    brain : surfer.Brain
        Plotted PySurfer brain

    Examples
    --------
    >>> import numpy as np
    >>> from netneurotools.datasets import fetch_cammoun2012, \
                                           fetch_schaefer2018
    >>> from netneurotools.plotting import plot_fsaverage

    Plotting with the Cammoun 2012 parcellation we specify `order='RL'` because
    many of the Lausanne connectomes have data for the right hemisphere before
    the left hemisphere.

    >>> values = np.random.rand(219)
    >>> scale = 'scale125'
    >>> cammoun = fetch_cammoun2012('fsaverage', verbose=False)[scale]
    >>> plot_fsaverage(values, order='RL',
    ...                lhannot=cammoun.lh, rhannot=cammoun.rh) # doctest: +SKIP

    Plotting with the Schaefer 2018 parcellation we can use the default
    parameter for `order`:

    >>> values = np.random.rand(400)
    >>> scale = '400Parcels7Networks'
    >>> schaefer = fetch_schaefer2018('fsaverage', verbose=False)[scale]
    >>> plot_fsaverage(values,
    ...                lhannot=schaefer.lh,
    ...                rhannot=schaefer.rh)  # doctest: +SKIP

    """

    subject_id, subjects_dir = _get_fs_subjid(subject_id, subjects_dir)

    # cast data to float (required for NaNs)
    data = np.asarray(data, dtype='float')

    order = order.lower()
    if order not in ('lr', 'rl'):
        raise ValueError('order must be either \'lr\' or \'rl\'')

    if mask is not None and len(mask) != len(data):
        raise ValueError('Provided mask must be the same length as data.')

    if vmin is None:
        vmin = np.nanpercentile(data, 2.5)
    if vmax is None:
        vmax = np.nanpercentile(data, 97.5)

    # parcels that should not be included in parcellation
    drop = FSIGNORE.copy()
    if noplot is not None:
        if isinstance(noplot, str):
            noplot = [noplot]
        drop += list(noplot)
    drop = _decode_list(drop)

    vtx_data = []
    for annot, hemi in zip((lhannot, rhannot), ('lh', 'rh')):
        # loads annotation data for hemisphere, including vertex `labels`!
        if not annot.startswith(os.path.abspath(os.sep)):
            annot = os.path.join(subjects_dir, subject_id, 'label', annot)
        labels, ctab, names = nib.freesurfer.read_annot(annot)
        names = _decode_list(names)

        # get appropriate data, accounting for hemispheric asymmetry
        currdrop = np.intersect1d(drop, names)
        if hemi == 'lh':
            if order == 'lr':
                split_id = len(names) - len(currdrop)
                ldata, rdata = np.split(data, [split_id])
                if mask is not None:
                    lmask, rmask = np.split(mask, [split_id])
            elif order == 'rl':
                split_id = len(data) - len(names) + len(currdrop)
                rdata, ldata = np.split(data, [split_id])
                if mask is not None:
                    rmask, lmask = np.split(mask, [split_id])
        hemidata = ldata if hemi == 'lh' else rdata

        # our `data` don't include the "ignored" parcels but our `labels` do,
        # so we need to account for that. find the label ids that correspond to
        # those and set them to NaN in the `data vector`
        inds = sorted([names.index(n) for n in currdrop])
        for i in inds:
            hemidata = np.insert(hemidata, i, np.nan)
        vtx = hemidata[labels]

        # let's also mask data, if necessary
        if mask is not None:
            maskdata = lmask if hemi == 'lh' else rmask
            maskdata = np.insert(maskdata, inds - np.arange(len(inds)), np.nan)
            vtx[maskdata[labels] > 0] = np.nan

        vtx_data.append(vtx)

    brain = plot_fsvertex(np.hstack(vtx_data), order='lr', mask=None,
                          subject_id=subject_id, subjects_dir=subjects_dir,
                          vmin=vmin, vmax=vmax, **kwargs)

    return brain


def plot_fsvertex(data, *, order='lr', surf='pial', views='lat',
                  vmin=None, vmax=None, center=None, mask=None,
                  colormap='viridis', colorbar=True, alpha=0.8,
                  label_fmt='%.2f', num_labels=3, size_per_view=500,
                  subject_id='fsaverage', subjects_dir=None, data_kws=None,
                  **kwargs):
    """
    Plots vertex-wise `data` to fsaverage brain.

    Parameters
    ----------
    data : (N,) array_like
        Data for `N` parcels as defined in `annot`
    order : {'lr', 'rl'}, optional
        Order of the hemispheres in the data vector. Default: 'lr'
    surf : str, optional
        Surface on which to plot data. Default: 'pial'
    views : str or list, optional
        Which views to plot of brain. Default: 'lat'
    vmin : float, optional
        Minimum value for colorbar. If not provided, a robust estimation will
        be used from values in `data`. Default: None
    vmax : float, optional
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
    subject : str, optional
        Subject ID to use; must be present in `subjects_dir`. Default:
        'fsaverage'
    data_kws : dict, optional
        Keyword arguments for Brain.add_data()

    Returns
    -------
    brain : surfer.Brain
        Plotted PySurfer brain
    """

    # hold off on imports until
    try:
        from surfer import Brain
    except ImportError:
        raise ImportError('Cannot use plot_fsaverage() if pysurfer is not '
                          'installed. Please install pysurfer and try again.')

    subject_id, subjects_dir = _get_fs_subjid(subject_id, subjects_dir)

    # cast data to float (required for NaNs)
    data = np.asarray(data, dtype='float')

    # handle data_kws if None
    if data_kws is None:
        data_kws = {}

    if mask is not None and len(mask) != len(data):
        raise ValueError('Provided mask must be the same length as data.')

    order = order.lower()
    if order not in ['lr', 'rl']:
        raise ValueError('Specified order must be either \'lr\' or \'rl\'')

    if vmin is None:
        vmin = np.nanpercentile(data, 2.5)
    if vmax is None:
        vmax = np.nanpercentile(data, 97.5)

    # set up brain views
    if not isinstance(views, (np.ndarray, list)):
        views = [views]

    # size of window will depend on # of views provided
    size = (size_per_view * 2, size_per_view * len(views))
    brain_kws = dict(background='white', size=size)
    brain_kws.update(**kwargs)
    brain = Brain(subject_id=subject_id, hemi='split', surf=surf,
                  subjects_dir=subjects_dir, views=views, **brain_kws)

    hemis = ('lh', 'rh') if order == 'lr' else ('rh', 'lh')
    for n, (hemi, vtx_data) in enumerate(zip(hemis, np.split(data, 2))):
        # let's mask data, if necessary
        if mask is not None:
            maskdata = np.asarray(np.split(mask, 2)[n], dtype=bool)
            vtx_data[maskdata] = np.nan

        # we don't want NaN values plotted so set a threshold if they exist
        thresh, nanmask = None, np.isnan(vtx_data)
        if np.any(nanmask) > 0:
            thresh = np.nanmin(vtx_data) - 1
            vtx_data[nanmask] = thresh
            thresh += 0.5

        # finally, add data to this hemisphere!
        brain.add_data(vtx_data, vmin, vmax, hemi=hemi, mid=center,
                       thresh=thresh, alpha=1.0, remove_existing=False,
                       colormap=colormap, colorbar=colorbar, verbose=False,
                       **data_kws)

        if alpha != 1.0:
            surf = brain.data_dict[hemi]['surfaces']
            for n, s in enumerate(surf):
                s.actor.property.opacity = alpha
                s.render()

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
            if num_labels is not None:
                for n, cm in enumerate(cmap):
                    cm.scalar_bar.number_of_labels = num_labels
                    surf[n].render()

    return brain


def plot_point_brain(data, coords, views=None, views_orientation='vertical',
                     views_size=(4, 2.4), cbar=False, robust=True, size=50,
                     **kwargs):
    """
    Plots `data` as a cloud of points in 3D space based on specified `coords`

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
