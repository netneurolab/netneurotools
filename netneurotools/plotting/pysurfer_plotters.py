"""Functions for pysurfer-based plotting."""

import os
import numpy as np
import nibabel as nib

from ..datasets import FREESURFER_IGNORE, _get_freesurfer_subjid

def plot_conte69(data, lhlabel, rhlabel, surf='midthickness',
                 vmin=None, vmax=None, colormap='viridis',
                 colorbar=True, num_labels=4, orientation='horizontal',
                 colorbartitle=None, backgroundcolor=(1, 1, 1),
                 foregroundcolor=(0, 0, 0), **kwargs):
    """
    Plot surface `data` on Conte69 Atlas.

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
    Plot surface `data` on a given fsLR32k atlas.

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
    from ..datasets import fetch_conte69, fetch_yerkes19
    try:
        from mayavi import mlab
    except ImportError:
        raise ImportError('Cannot use plot_fslr() if mayavi is not '
                          'installed. Please install mayavi and try again.') from None

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
                         .format(surf_type)) from None

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





def plot_fsaverage(data, *, lhannot, rhannot, order='lr', mask=None,
                   noplot=None, subject_id='fsaverage', subjects_dir=None,
                   vmin=None, vmax=None, **kwargs):
    """
    Plot `data` to fsaverage brain using `annot` as parcellation.

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
    def _decode_list(vals):
        """List decoder."""
        return [val.decode() if hasattr(val, 'decode') else val for val in vals]

    subject_id, subjects_dir = _get_freesurfer_subjid(subject_id, subjects_dir)

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
    drop = FREESURFER_IGNORE.copy()
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
    Plot vertex-wise `data` to fsaverage brain.

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
                          'installed. Please install pysurfer and try again.') from None

    subject_id, subjects_dir = _get_freesurfer_subjid(subject_id, subjects_dir)

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
            for _, s in enumerate(surf):
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