"""Functions for pyvista-based plotting."""

from pathlib import Path

import nibabel as nib
import numpy as np

import matplotlib as mpl
import matplotlib.colors as mcolors

try:
    import pyvista as pv
except ImportError:
    _has_pyvista = False
else:
    _has_pyvista = True

from netneurotools.datasets import (
    fetch_civet_curated,
    fetch_fsaverage_curated,
    fetch_fslr_curated,
)

from netneurotools.datasets.fetch_template import (
    _fetch_subcortex_surface
)


def _pv_fetch_template(template, surf="inflated", data_dir=None, verbose=0):
    if template in ["fsaverage", "fsaverage6", "fsaverage5", "fsaverage4"]:
        _fetch_curr_tpl = fetch_fsaverage_curated
    elif template in ["fslr4k", "fslr8k", "fslr32k", "fslr164k"]:
        _fetch_curr_tpl = fetch_fslr_curated
    elif template in ["civet41k", "civet164k"]:
        _fetch_curr_tpl = fetch_civet_curated
    else:
        raise ValueError(f"Unknown template: {template}")

    curr_tpl_surf = _fetch_curr_tpl(
        version=template, data_dir=data_dir, verbose=verbose
    )[surf]

    return curr_tpl_surf


def _pv_load_surface(template, surf="inflated", hemi=None, data_dir=None, verbose=0):
    curr_tpl_surf = _pv_fetch_template(
        template=template, surf=surf, data_dir=data_dir, verbose=verbose
    )

    def _gifti_to_polydata(gifti_file):
        vertices, faces = nib.load(gifti_file).agg_data()
        return pv.PolyData(
            vertices, np.c_[np.ones((faces.shape[0],), dtype=int) * 3, faces]
        )

    if hemi == "L":
        return _gifti_to_polydata(curr_tpl_surf.L)
    elif hemi == "R":
        return _gifti_to_polydata(curr_tpl_surf.R)
    else:
        return (
            _gifti_to_polydata(curr_tpl_surf.L),
            _gifti_to_polydata(curr_tpl_surf.R),
        )


def _mask_medial_wall(data, template, hemi=None, data_dir=None, verbose=0):
    curr_medial = _pv_fetch_template(
        template=template, surf="medial", data_dir=data_dir, verbose=verbose
    )
    if isinstance(data, tuple):
        curr_medial_data = (
            nib.load(curr_medial.L).agg_data(),
            nib.load(curr_medial.R).agg_data(),
        )
        # Convert to float to support NaN masking for missing/masked vertices
        ret_L = data[0].astype(float)
        ret_R = data[1].astype(float)
        ret_L[np.where(1 - curr_medial_data[0])] = np.nan
        ret_R[np.where(1 - curr_medial_data[1])] = np.nan
        ret = (ret_L, ret_R)
    else:
        if hemi == "L":
            curr_medial_data = nib.load(curr_medial.L).agg_data()
        elif hemi == "R":
            curr_medial_data = nib.load(curr_medial.R).agg_data()
        else:
            curr_medial_data = np.concatenate(
                [
                    nib.load(curr_medial.L).agg_data(),
                    nib.load(curr_medial.R).agg_data(),
                ],
                axis=1,
            )
        ret = data.copy()
        ret[np.where(1 - curr_medial_data)] = np.nan
    return ret


def _pv_update_settings(
    panel_size, plotter_shape, scalars,
    cmap, _vmin, _vmax, cbar_title,
    lighting_style,
    jupyter_backend,
    plotter_kws, mesh_kws, cbar_kws, silhouette_kws
):
    plotter_settings = {
        "border": False,
        "lighting": "three lights",
    }

    plotter_settings["window_size"] = (
        panel_size[0] * plotter_shape[1],
        panel_size[1] * plotter_shape[0]
    )

    if jupyter_backend is not None:
        plotter_settings.update(dict(notebook=True, off_screen=True))

    mesh_settings = {
        "smooth_shading": True,
        "show_scalar_bar": False,
    }

    mesh_settings.update(
        dict(
            scalars=scalars,
            cmap=cmap,
            clim=(_vmin, _vmax),
        )
    )

    lighting_style_keys = [
        "ambient",
        "diffuse",
        "specular",
        "specular_power"
    ]
    lighting_style_presets = {
        "metallic": [0.1, 0.3, 1.0, 10],
        "plastic": [0.3, 0.4, 0.3, 5],
        "shiny": [0.2, 0.6, 0.8, 50],
        "glossy": [0.1, 0.7, 0.9, 90],
        "ambient": [0.8, 0.1, 0.0, 1],
        "plain": [0.1, 1.0, 0.05, 5],
    }

    if lighting_style in ["default", "lightkit"]:
        mesh_settings["lighting"] = "light kit"
    elif lighting_style == "threelights":
        mesh_settings["lighting"] = "three lights"
    elif lighting_style in lighting_style_presets.keys():
        mesh_settings.update(
            {
                k: v
                for k, v in zip(
                    lighting_style_keys, lighting_style_presets[lighting_style]
                )
            }
        )
        mesh_settings["lighting"] = "light kit"
    elif lighting_style == "none":
        plotter_settings["lighting"] = "none"
        mesh_settings["lighting"] = False
    else:
        raise ValueError(f"Unknown lighting style: {lighting_style}")

    cbar_settings = dict(
        title=cbar_title,
        n_labels=2,
        label_font_size=20,
        title_font_size=24,
        font_family="arial",
        height=0.15,
    )

    silhouette_settings = {}

    if plotter_kws is not None:
        plotter_settings.update(plotter_kws)
    if mesh_kws is not None:
        mesh_settings.update(mesh_kws)
    if cbar_kws is not None:
        cbar_settings.update(cbar_kws)
    if silhouette_kws is not None:
        silhouette_settings.update(silhouette_kws)

    return plotter_settings, mesh_settings, cbar_settings, silhouette_settings


def _pv_get_plotter_shape(hemi, layout):
    shapes = {
        ("default", "both"): (2, 2),
        ("default", "single_hemi"): (1, 2),
        ("single", "both"): (1, 1),
        ("single", "single_hemi"): (1, 1),
        ("row", "both"): (1, 4),
        ("row", "single_hemi"): (1, 2),
        ("column", "both"): (4, 1),
        ("column", "single_hemi"): (2, 1),
    }
    key = (layout, "both" if hemi == "both" else "single_hemi")
    if key not in shapes:
        raise ValueError(f"Unknown layout: {layout}")
    return shapes[key]


def _pv_add_colorbar(
    pl,
    layout,
    _vmin,
    _vmax,
    cmap,
    cbar_settings
):
    if layout == "default":
        pl.subplot(1, 1)
    elif layout == "row":
        pl.subplot(0, 3)
    elif layout == "column":
        pl.subplot(3, 0)
    elif layout == "single":
        pl.subplot(0, 0)

    _mesh = pv.PolyData(np.zeros((2, 3)))
    _mesh['data'] = (_vmin, _vmax)

    actor = pl.add_mesh(
        _mesh, scalars=None, show_scalar_bar=False,
        cmap=cmap, clim=(_vmin, _vmax)
    )
    actor.visibility = False

    cbar = pl.add_scalar_bar(mapper=actor.mapper, **cbar_settings)
    cbar.GetLabelTextProperty().SetItalic(True)


def _pv_save_fig(pl, save_fig):
    _fname = Path(save_fig)
    raster_formats = {".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff"}
    vector_formats = {".svg", ".eps", ".ps", ".pdf", ".tex"}

    if _fname.suffix in raster_formats:
        pl.screenshot(_fname, return_img=False)
    elif _fname.suffix in vector_formats:
        pl.save_graphic(_fname)
    else:
        raise ValueError(f"Unknown file format: {save_fig}")


def pv_plot_surface(
    vertex_data,
    template,
    surf="inflated",
    hemi="both",
    layout="default",
    mask_medial=True,
    cmap="viridis",
    clim=None,
    panel_size=(700, 500),
    zoom_ratio=1.25,
    show_colorbar=True,
    show_silhouette=False,
    cbar_title=None,
    show_plot=True,
    jupyter_backend="static",
    lighting_style="default",
    save_fig=None,
    plotter_kws=None,
    mesh_kws=None,
    cbar_kws=None,
    silhouette_kws=None,
    data_dir=None,
    verbose=0,
):
    """
    Plot surface data using PyVista.

    This function provides a flexible interface for visualizing cortical surface
    data on standard neuroimaging templates. It supports multiple hemispheres,
    layouts, lighting styles, and customization options.

    Parameters
    ----------
    vertex_data : array-like or tuple of array-like
        Data array(s) to be plotted on the surface. If `hemi` is "both", this
        should be a tuple of two arrays (left, right) or a single concatenated
        array. For single hemisphere, provide a single array matching the number
        of vertices in that hemisphere.
    template : str
        Template to use for plotting. Options include 'fsaverage', 'fsaverage6',
        'fsaverage5', 'fsaverage4', 'fslr4k', 'fslr8k', 'fslr32k', 'fslr164k',
        'civet41k', 'civet164k'.
    surf : str, optional
        Surface type to plot. Available options depend on template:

        - fsaverage templates: 'midthickness', 'pial', 'white', 'inflated', 'sphere'
        - fslr templates: 'midthickness', 'pial', 'white', 'inflated', 'veryinflated', 'sphere'
        - civet templates: 'midthickness', 'white', 'inflated'

        Default is 'inflated'.
    hemi : str, optional
        Hemisphere to plot. Options: 'L' (left), 'R' (right), 'both'.
        Default is 'both'.
    layout : str, optional
        Layout of the plot panels:

        - 'default': 2x2 grid for both hemispheres, 1x2 for single hemisphere
        - 'single': Single panel (useful for custom views)
        - 'row': Horizontal arrangement of all views
        - 'column': Vertical arrangement of all views

        Default is 'default'.
    mask_medial : bool, optional
        Whether to mask the medial wall (set to NaN). Only applies to templates
        with medial wall annotations. Default is True.
    cmap : str, optional
        Matplotlib colormap name. Default is 'viridis'.
    clim : tuple of float, optional
        Colorbar limits as (vmin, vmax). If None, will be set to 2.5th and
        97.5th percentiles of the data. Default is None.
    panel_size : tuple of int, optional
        Size of each panel in pixels as (width, height). Default is (700, 500).
    zoom_ratio : float, optional
        Camera zoom level. Values > 1.0 zoom in, < 1.0 zoom out.
        Default is 1.25.
    show_colorbar : bool, optional
        Whether to display the colorbar. Default is True.
    cbar_title : str, optional
        Title text for the colorbar. Default is None.
    show_plot : bool, optional
        Whether to display the plot immediately. Set to False to return the
        plotter object for further customization. Default is True.
    jupyter_backend : str, optional
        Backend for Jupyter notebook rendering. See `PyVista documentation
        <https://docs.pyvista.org/user-guide/jupyter/index.html#pyvista.set_jupyter_backend>`_
        for available options ('html', 'static', 'trame', etc.).
        Set to None for non-notebook environments. Default is 'static'.
    lighting_style : str, optional
        Lighting style preset:

        - 'default', 'lightkit': Standard three-point lighting
        - 'threelights': Alternative three-light setup
        - 'metallic', 'plastic', 'shiny', 'glossy': Material presets
        - 'ambient', 'plain': Flat lighting styles

        Default is 'default'.
    save_fig : str or Path, optional
        Path to save the figure. Supported formats: .png, .jpeg, .jpg, .bmp,
        .tif, .tiff (raster); .svg, .eps, .ps, .pdf, .tex (vector).
        Default is None (no save).

    Returns
    -------
    pl : :class:`pyvista.Plotter`
        PyVista plotter object. Can be further customized before calling
        `pl.show()` if `show_plot=False`.

    Other Parameters
    ----------------
    plotter_kws : dict, optional
        Additional keyword arguments to pass to
        :class:`pyvista.Plotter`. Default is None.
    mesh_kws : dict, optional
        Additional keyword arguments to pass to
        :meth:`pyvista.Plotter.add_mesh`. Default is None.
    cbar_kws : dict, optional
        Additional keyword arguments to pass to
        :meth:`pyvista.Plotter.add_scalar_bar`. Default is None.
    silhouette_kws : dict, optional
        Additional keyword arguments to pass to
        :meth:`pyvista.Plotter.add_silhouette`. Only used when
        `lighting_style='silhouette'`. Default is None.
    data_dir : str or Path, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 0

    Notes
    -----
    **Template and surface compatibility:**

    Not all surface types are available for all templates. The function will
    automatically fetch the appropriate template data from neuromaps or use
    locally cached data.

    **Layouts:**

    - 'default': Shows medial and lateral views for each hemisphere
    - 'single': Shows only one view (useful for custom camera angles)
    - 'row'/'column': Linear arrangements of all standard views

    **Data format:**

    The `vertex_data` array(s) must match the number of vertices in the surface
    template. For example, fsaverage5 has 10,242 vertices per hemisphere.

    When `hemi='both'`, vertex_data can be:

    - A tuple/list: (left_data, right_data)
    - A concatenated array: np.concatenate([left_data, right_data])

    The function automatically handles data splitting based on template vertex
    counts.

    **Parcellated data:**

    If you have parcellated/regional data rather than vertex-wise data, use
    :func:`netneurotools.interface.parcels_to_vertices` to convert it to
    vertex-level data before plotting. Always verify the data before and after
    transformation to ensure correct mapping.

    **Lighting styles:**

    Different lighting presets affect surface appearance through ambient,
    diffuse, specular, and specular_power parameters:

    - Metallic: Low ambient (0.1), high specular (1.0)
    - Plastic: Balanced properties, moderate specular (0.3)
    - Shiny: High specular (0.8), high specular_power (50)

    **Jupyter notebooks:**

    When using in Jupyter, the function automatically sets `notebook=True` and
    `off_screen=True` for proper rendering.

    There can be various issues when plotting in Jupyter notebooks depending on
    your environment. For troubleshooting and detailed configuration options, see:

    - `Trame Jupyter Guide <https://kitware.github.io/trame/guide/jupyter/intro.html>`_
    - `PyVista Jupyter Documentation
      <https://docs.pyvista.org/user-guide/jupyter/>`_

    **Backend selection:**

    Choose the appropriate `jupyter_backend` for your use case:

    - `'trame'`: Best performance and interactivity (recommended)
    - `'html'`: Good interactivity, works in most environments
    - `'static'`: No interactivity but reliable fallback option

    If trame does not work in your environment, try html. The static option
    should always work as a last resort.

    **Customization with keyword arguments:**

    The `plotter_kws`, `mesh_kws`, `cbar_kws`, and `silhouette_kws` parameters
    allow flexible overriding of default settings. For example:

    - `plotter_kws={'window_size': (2000, 1500)}` for higher resolution
    - `mesh_kws={'smooth_shading': False}` to disable smooth shading
    - `cbar_kws={'n_labels': 5}` for more colorbar labels
    - `silhouette_kws={'feature_angle': 30}` to adjust edge detection sensitivity

    Examples
    --------
    **Basic usage:**

    Plot random data on fsaverage5 pial surface:

    >>> from netneurotools.plotting import pv_plot_surface  # doctest: +SKIP
    >>> import numpy as np  # doctest: +SKIP
    >>> data_L = np.random.random((10242,))  # doctest: +SKIP
    >>> data_R = np.random.random((10242,))  # doctest: +SKIP
    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="pial"
    ... )

    **Available template/surface combinations:**

    >>> # fsaverage templates (all densities)
    >>> templates = ["fsaverage", "fsaverage6", "fsaverage5", "fsaverage4"]  # doctest: +SKIP
    >>> surfaces = ["midthickness", "pial", "white", "inflated", "sphere"]  # doctest: +SKIP
    >>>
    >>> # fslr templates
    >>> templates = ["fslr4k", "fslr8k", "fslr32k", "fslr164k"]  # doctest: +SKIP
    >>> surfaces = ["midthickness", "pial", "white", "inflated",  # doctest: +SKIP
    ...             "veryinflated", "sphere"]
    >>>
    >>> # civet templates
    >>> templates = ["civet41k", "civet164k"]  # doctest: +SKIP
    >>> surfaces = ["midthickness", "white", "inflated"]  # doctest: +SKIP

    **Different layouts:**

    Compare all layout options:

    >>> for layout in ["default", "single", "row", "column"]:  # doctest: +SKIP
    ...     pl = pv_plot_surface(
    ...         (data_L, data_R),
    ...         template="fsaverage5",
    ...         surf="inflated",
    ...         layout=layout,
    ...         cbar_title=f"Layout: {layout}"
    ...     )

    **Adjusting zoom:**

    Control the camera zoom level:

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="inflated",
    ...     zoom_ratio=1.7,  # Closer view
    ... )

    **Colorbar control:**

    Customize colorbar display and title:

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="inflated",
    ...     show_colorbar=True,
    ...     cbar_title="Activation (z-score)",
    ... )

    Hide the colorbar:

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="inflated",
    ...     show_colorbar=False,
    ... )

    **Colormap and limits:**

    Use different colormaps and set explicit color limits:

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="inflated",
    ...     cmap="RdBu_r",  # Reverse red-blue colormap
    ...     clim=(-3, 3),   # Symmetric limits
    ... )

    Sequential colormap for positive-only data:

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="inflated",
    ...     cmap="plasma",
    ...     clim=(0, 1),
    ... )

    **Saving figures:**

    Save as high-resolution PNG:

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="inflated",
    ...     save_fig="brain_plot.png",
    ... )

    Save as vector graphics (SVG):

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="inflated",
    ...     save_fig="brain_plot.svg",
    ... )

    **Lighting styles:**

    Explore different lighting presets:

    >>> for style in ["metallic", "plastic", "shiny", "glossy"]:  # doctest: +SKIP
    ...     pl = pv_plot_surface(
    ...         (data_L, data_R),
    ...         template="fsaverage5",
    ...         surf="inflated",
    ...         lighting_style=style,
    ...         cbar_title=f"Style: {style}",
    ...         save_fig=f"brain_{style}.png",
    ...     )

    Silhouette style for presentations (often needs tuning):

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="inflated",
    ...     show_silhouette=True,  # High-contrast edges
    ...     cmap="coolwarm",
    ...     silhouette_kws={'feature_angle': 30, 'color': 'black'},
    ... )

    **Customizing with keyword arguments:**

    Override default settings for higher resolution figures:

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="inflated",
    ...     plotter_kws={'window_size': (2000, 1500)},  # Higher resolution
    ... )

    Disable smooth shading for sharper vertex transitions:

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="inflated",
    ...     mesh_kws={'smooth_shading': False},
    ... )

    **Advanced customization:**

    Combine multiple options:

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fslr32k",
    ...     surf="veryinflated",
    ...     layout="row",
    ...     mask_medial=True,
    ...     cmap="viridis",
    ...     clim=(0.2, 0.8),
    ...     zoom_ratio=1.5,
    ...     show_colorbar=True,
    ...     cbar_title="Correlation",
    ...     lighting_style="shiny",
    ...     save_fig="publication_figure.png",
    ...     jupyter_backend=None,  # For script usage
    ... )

    Use plotter object for further customization:

    >>> pl = pv_plot_surface(  # doctest: +SKIP
    ...     (data_L, data_R),
    ...     template="fsaverage5",
    ...     surf="inflated",
    ...     show_plot=False,  # Don't show yet
    ... )
    >>> # Add custom annotations, lights, etc.
    >>> pl.add_text("Custom Title", position="upper_edge")  # doctest: +SKIP
    >>> pl.show()  # doctest: +SKIP
    """  # noqa: E501
    if not _has_pyvista:
        raise ImportError("PyVista is required for this function")

    if hemi not in ["L", "R", "both"]:
        raise ValueError(f"Unknown hemi: {hemi}")

    # Prepare and validate surface data for both or single hemisphere
    if hemi == "both":  # Process both hemispheres
        surf_pair = _pv_load_surface(
            template=template, surf=surf, data_dir=data_dir, verbose=verbose
        )
        if len(vertex_data) == 2:  # Input is tuple/list of (left_data, right_data)
            # Validate data length matches number of vertices for each hemisphere
            if not all(len(vertex_data[i]) == surf_pair[i].n_points for i in range(2)):
                raise ValueError("Data length mismatch")
        else:  # Input is single concatenated array
            # Validate total data length matches combined vertex count
            if len(vertex_data) != surf_pair[0].n_points + surf_pair[1].n_points:
                raise ValueError("Data length mismatch")
            # Split concatenated array into left and right hemispheres
            vertex_data = (
                vertex_data[: surf_pair[0].n_points],
                vertex_data[surf_pair[0].n_points :],
            )

        if mask_medial:
            vertex_data = _mask_medial_wall(
                vertex_data, template, hemi=None, data_dir=data_dir, verbose=verbose
            )
        surf_pair[0].point_data["vertex_data"] = vertex_data[0]
        surf_pair[1].point_data["vertex_data"] = vertex_data[1]
    else:
        # Process single hemisphere with validation
        surf = _pv_load_surface(
            template=template, surf=surf, hemi=hemi, data_dir=data_dir, verbose=verbose
        )
        if len(vertex_data) != surf.n_points:
            raise ValueError("Data length mismatch")

        if mask_medial:
            vertex_data = _mask_medial_wall(
                vertex_data, template, hemi=hemi, data_dir=data_dir, verbose=verbose
            )
        surf.point_data["vertex_data"] = vertex_data

    # Determine grid layout based on number of hemispheres and layout preference
    plotter_shape = _pv_get_plotter_shape(hemi, layout)

    # Set colorbar scale: use provided limits or calculate from data percentiles
    if clim is not None:
        _vmin, _vmax = clim
    else:
        if len(vertex_data) == 2:
            _values = np.r_[vertex_data[0], vertex_data[1]]
        else:
            _values = vertex_data
        # Use 2.5th and 97.5th percentiles to handle outliers
        _vmin, _vmax = np.nanpercentile(_values, [2.5, 97.5])

    plotter_settings, mesh_settings, cbar_settings, silhouette_settings = \
        _pv_update_settings(
            panel_size=panel_size,
            plotter_shape=plotter_shape,
            scalars="vertex_data",
            cmap=cmap,
            _vmin=_vmin,
            _vmax=_vmax,
            cbar_title=cbar_title,
            lighting_style=lighting_style,
            jupyter_backend=jupyter_backend,
            plotter_kws=plotter_kws,
            mesh_kws=mesh_kws,
            cbar_kws=cbar_kws,
            silhouette_kws=silhouette_kws
        )

    pl = pv.Plotter(shape=plotter_shape, **plotter_settings)

    if layout == "single":  # Single panel view
        if hemi == "both":
            _surf = surf_pair[0]
            _view_flip = True
        else:
            _surf = surf
            if hemi == "L":
                _view_flip = True
            else:
                _view_flip = False

        pl.subplot(0, 0)
        pl.add_mesh(_surf, **mesh_settings)
        pl.view_yz(negative=_view_flip)
        pl.zoom_camera(zoom_ratio)
        if show_silhouette:
            pl.add_silhouette(_surf, **silhouette_settings)
    else:  # Multi-panel layout with multiple views
        if hemi == "both":  # Display 4 panels: 2 views per hemisphere
            if layout == "default":
                _pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
            elif layout == "row":
                _pos = [(0, 0), (0, 3), (0, 1), (0, 2)]
            elif layout == "column":
                _pos = [(0, 0), (2, 0), (1, 0), (3, 0)]
            else:
                raise ValueError(f"Unknown layout: {layout}")

            _surf_list = [
                surf_pair[0], surf_pair[1], surf_pair[0], surf_pair[1]
            ]
            _view_flip_list = [True, False, False, True]

            for _xy, _surf, _view_flip in zip(_pos, _surf_list, _view_flip_list):
                pl.subplot(*_xy)
                pl.add_mesh(_surf, **mesh_settings)
                pl.view_yz(negative=_view_flip)
                pl.zoom_camera(zoom_ratio)
                if show_silhouette:
                    pl.add_silhouette(_surf, **silhouette_settings)
        else:  # Display 2 panels: medial and lateral views of single hemisphere
            if layout == "default":
                _pos = [(0, 0), (0, 1)]
            elif layout == "row":
                _pos = [(0, 0), (0, 1)]
            elif layout == "column":
                _pos = [(0, 0), (1, 0)]
            else:
                raise ValueError(f"Unknown layout: {layout}")

            _surf_list = [surf, surf]
            if hemi == "L":
                _view_flip_list = [True, False]
            else:
                _view_flip_list = [False, True]

            for _xy, _surf, _view_flip in zip(_pos, _surf_list, _view_flip_list):
                pl.subplot(*_xy)
                pl.add_mesh(_surf, **mesh_settings)
                pl.view_yz(negative=_view_flip)
                pl.zoom_camera(zoom_ratio)
                if show_silhouette:
                    pl.add_silhouette(_surf, **silhouette_settings)

    if show_colorbar:
        _pv_add_colorbar(
            pl=pl,
            layout=layout,
            _vmin=_vmin,
            _vmax=_vmax,
            cmap=cmap,
            cbar_settings=cbar_settings
        )

    if show_plot:
        if jupyter_backend is not None:
            pl.show(jupyter_backend=jupyter_backend)
        else:
            pl.show()

    if save_fig is not None:
        _pv_save_fig(pl, save_fig)

    return pl


def _pv_make_subcortex_surfaces(
    atlas, include_keys, custom_params=None
):
    """
    Generate surface meshes from volumetric subcortical atlas data.

    This function converts volumetric segmentation data (e.g., NIfTI files) into
    3D surface meshes suitable for visualization with PyVista. It uses a pipeline
    of Gaussian smoothing, marching cubes algorithm, and Taubin smoothing to
    create high-quality surface representations.

    .. warning::
        This function is considered experimental and may be subject to changes anytime.

    Parameters
    ----------
    atlas : str or Path
        Path to the volumetric atlas file (e.g., NIfTI format) containing
        integer-labeled regions.
    include_keys : list of int
        List of integer region identifiers to convert to surface meshes.
        These should match the integer labels in the atlas file.
    custom_params : dict, optional
        Custom parameters to override defaults for the meshification process.
        Can contain:

        - 'gaussian_filter': dict with 'sigma' (float) and 'threshold' (float)
        - 'taubin_smoothing': dict with 'lamb' (float), 'nu' (float), and
          'iterations' (int)

        Default is None.

    Returns
    -------
    multiblock : :class:`pyvista.MultiBlock`
        PyVista MultiBlock object containing surface meshes for each region.
        Keys are string versions of the input region identifiers.

    Notes
    -----
    **Meshification pipeline:**

    The function converts volumetric data to surface meshes through three steps:

    1. **Gaussian smoothing**: Applies a 3D Gaussian filter to smooth the binary
       mask for each region. The `sigma` parameter controls the smoothing strength.
       Higher values create smoother, more rounded surfaces but may lose fine
       details. Lower values preserve details but may result in blocky surfaces.

    2. **Marching cubes**: Extracts an isosurface from the smoothed volume using
       the marching cubes algorithm. The `threshold` parameter determines the
       isosurface level. Values around 0.5 work well for binary masks after
       Gaussian smoothing.

    3. **Taubin smoothing**: Applies mesh smoothing to reduce surface artifacts
       while preserving volume. The `lamb` parameter controls the amount of
       smoothing (positive shrinking step), `nu` controls the inverse step
       (negative expansion), and `iterations` determines how many smoothing
       passes to apply. More iterations create smoother surfaces but may
       over-smooth fine features.

    **Default parameters:**

    - Gaussian filter: sigma=1.0, threshold=0.5
    - Taubin smoothing: lamb=0.75, nu=0.6, iterations=100

    These defaults work well for standard 1mm isotropic MNI space atlases.
    Adjust parameters based on your atlas resolution and desired smoothness.

    Examples
    --------
    **Basic usage:**

    Generate surfaces for specific regions from a FreeSurfer aseg atlas:

    >>> from netneurotools.plotting import _pv_make_subcortex_surfaces  # doctest: +SKIP
    >>> surfaces = _pv_make_subcortex_surfaces(  # doctest: +SKIP
    ...     atlas="/path/to/aseg.nii.gz",
    ...     include_keys=[10, 11, 12, 13],  # thalamus, caudate, putamen, pallidum
    ... )

    **Custom smoothing parameters:**

    Create smoother surfaces by increasing Gaussian sigma:

    >>> custom_params = {  # doctest: +SKIP
    ...     "gaussian_filter": {"sigma": 1.5, "threshold": 0.5},
    ...     "taubin_smoothing": {"lamb": 0.75, "nu": 0.6, "iterations": 100}
    ... }
    >>> surfaces = _pv_make_subcortex_surfaces(  # doctest: +SKIP
    ...     atlas="/path/to/atlas.nii.gz",
    ...     include_keys=[1, 2, 3],
    ...     custom_params=custom_params
    ... )

    **Access individual surfaces:**

    The returned MultiBlock can be indexed by string keys:

    >>> surfaces = _pv_make_subcortex_surfaces(  # doctest: +SKIP
    ...     atlas="/path/to/atlas.nii.gz",
    ...     include_keys=[10, 11]
    ... )
    >>> thalamus_mesh = surfaces['10']  # doctest: +SKIP
    >>> caudate_mesh = surfaces['11']  # doctest: +SKIP

    **Saving generated surfaces:**

    The MultiBlock object can be saved to a file for later use:

    >>> surfaces.save("subcortex_surfaces.vtm")  # doctest: +SKIP

    Note that the .vtm format preserves the MultiBlock structure.
    It consists of an XML file (.vtm) and associated vtk mesh files (.vtp)
    stored in a folder with the same name as the .vtm file.

    To load the saved surfaces later:
    >>> import pyvista as pv  # doctest: +SKIP
    >>> loaded_surfaces = pv.read("subcortex_surfaces.vtm")  # doctest: +SKIP

    **Use with pv_plot_subcortex:**

    Generate custom surfaces and visualize them:

    >>> from netneurotools.plotting import pv_plot_subcortex  # doctest: +SKIP
    >>> surfaces = _pv_make_subcortex_surfaces(  # doctest: +SKIP
    ...     atlas="/path/to/custom_atlas.nii.gz",
    ...     include_keys=['1', '2', '3', '4']
    ... )
    >>> parcel_data = {'1': 0.5, '2': 0.7, '3': 0.6, '4': 0.8}  # doctest: +SKIP
    >>> pl = pv_plot_subcortex(  # doctest: +SKIP
    ...     parcel_data,
    ...     template="custom",
    ...     custom_surfaces=surfaces
    ... )
    """  # noqa: E501
    try:
        from trimesh.voxel.ops import matrix_to_marching_cubes
        from trimesh.smoothing import filter_taubin
        from scipy.ndimage import gaussian_filter
    except ImportError:
        raise ImportError(
            "trimesh is required for this function"
        ) from None

    default_params = {
        "gaussian_filter": {
            "sigma": 1.0,
            "threshold": 0.5,
        },
        "taubin_smoothing": {
            "lamb": 0.75,
            "nu": 0.6,
            "iterations": 100,
        }
    }
    default_params.update(custom_params or {})

    atlas_fdata = nib.load(atlas).get_fdata()

    def region_to_mesh(mask):
        return filter_taubin(
            matrix_to_marching_cubes(
                gaussian_filter(
                    mask.astype(float),
                    sigma=default_params["gaussian_filter"]["sigma"]
                ) > default_params["gaussian_filter"]["threshold"]
            ),
            lamb=default_params["taubin_smoothing"]["lamb"],
            nu=default_params["taubin_smoothing"]["nu"],
            iterations=default_params["taubin_smoothing"]["iterations"]
        )
    # Note: input include_keys are integers from the atlas volume, but output keys
    # are converted to strings for PyVista MultiBlock compatibility and consistent
    # dictionary access patterns across different template formats
    multiblock = pv.MultiBlock()
    for k in include_keys:
        multiblock[str(k)] = pv.wrap(region_to_mesh(atlas_fdata == int(k)))

    return multiblock


def _pv_load_subcortex_surfaces(
    template, include_keys, force_fetch=False, data_dir=None, verbose=0
):
    # Load pre-computed subcortical surface meshes from cache or download if needed
    if template in ["aseg", "tianS1", "tianS2", "tianS3", "tianS4"]:
        surf_vtm = _fetch_subcortex_surface(
            force=force_fetch, data_dir=data_dir, verbose=verbose
        )[template]
    else:
        raise ValueError(f"Unknown template: {template}")

    multiblock = pv.read(surf_vtm)
    meshes = [multiblock[name] for name in include_keys]
    return meshes


def pv_plot_subcortex(
    parcel_data,
    template,
    include_keys=None,
    custom_surfaces=None,
    hemi="both",
    layout="default",
    cmap="viridis",
    clim=None,
    panel_size=(500, 400),
    zoom_ratio=1.4,
    show_colorbar=True,
    show_silhouette=False,
    parallel_projection=True,
    cbar_title=None,
    show_plot=True,
    jupyter_backend="static",
    lighting_style="default",
    save_fig=None,
    plotter_kws=None,
    mesh_kws=None,
    cbar_kws=None,
    silhouette_kws=None,
    force_fetch=False,
    data_dir=None,
    verbose=0,
):
    """
    Plot subcortical data using PyVista.

    This function provides a flexible interface for visualizing parcellated
    subcortical data on standard neuroimaging atlases. It supports custom
    surfaces, multiple layouts, and various customization options for
    publication-quality visualizations.

    Parameters
    ----------
    parcel_data : dict
        Dictionary mapping region identifiers to data values. Keys should match
        the region identifiers in the selected template atlas (e.g., '10' for
        a specific region in aseg template).
    template : str
        Atlas template to use for subcortical visualization. Options:

        - 'aseg': FreeSurfer automatic segmentation
        - 'tianS1', 'tianS2', 'tianS3', 'tianS4': Tian et al. subcortical atlas
        - 'custom': User-provided custom surfaces (requires `custom_surfaces`)

    include_keys : list or tuple of lists, optional
        Region identifiers to include in the visualization. If `hemi` is "both",
        this should be a tuple of two lists (left, right). If None, will use all
        keys from `parcel_data`. Default is None.
    custom_surfaces : dict, optional
        Dictionary mapping region identifiers (as strings) to PyVista mesh objects.
        Only used when `template='custom'`. Should be generated with
        :func:`_pv_make_subcortex_surfaces` or similar. Default is None.
    hemi : str, optional
        Hemisphere to plot. Options: 'L' (left), 'R' (right), 'both'.
        Default is 'both'.
    layout : str, optional
        Layout of the plot panels:

        - 'default': 2x2 grid for both hemispheres, 1x2 for single hemisphere
        - 'single': Single panel (useful for custom views)
        - 'row': Horizontal arrangement of all views
        - 'column': Vertical arrangement of all views

        Default is 'default'.
    cmap : str, optional
        Matplotlib colormap name. Default is 'viridis'.
    clim : tuple of float, optional
        Colorbar limits as (vmin, vmax). If None, will be set to 2.5th and
        97.5th percentiles of the data. Default is None.
    panel_size : tuple of int, optional
        Size of each panel in pixels as (width, height). Default is (500, 400).
    zoom_ratio : float, optional
        Camera zoom level. Values > 1.0 zoom in, < 1.0 zoom out.
        Default is 1.4.
    show_colorbar : bool, optional
        Whether to display the colorbar. Default is True.
    show_silhouette : bool, optional
        Whether to add silhouette edges to the meshes for enhanced visibility.
        Default is False.
    parallel_projection : bool, optional
        Whether to use parallel projection for the camera. Default is True.
    cbar_title : str, optional
        Title text for the colorbar. Default is None.
    show_plot : bool, optional
        Whether to display the plot immediately. Set to False to return the
        plotter object for further customization. Default is True.
    jupyter_backend : str, optional
        Backend for Jupyter notebook rendering. See `PyVista documentation
        <https://docs.pyvista.org/user-guide/jupyter/index.html#pyvista.set_jupyter_backend>`_
        for available options ('html', 'static', 'trame', etc.).
        Set to None for non-notebook environments. Default is 'static'.
    lighting_style : str, optional
        Lighting style preset:

        - 'default', 'lightkit': Standard three-point lighting
        - 'threelights': Alternative three-light setup
        - 'metallic', 'plastic', 'shiny', 'glossy': Material presets
        - 'ambient', 'plain': Flat lighting styles

        Default is 'default'.
    save_fig : str or Path, optional
        Path to save the figure. Supported formats: .png, .jpeg, .jpg, .bmp,
        .tif, .tiff (raster); .svg, .eps, .ps, .pdf, .tex (vector).
        Default is None (no save).

    Returns
    -------
    pl : :class:`pyvista.Plotter`
        PyVista plotter object. Can be further customized before calling
        `pl.show()` if `show_plot=False`.

    Other Parameters
    ----------------
    plotter_kws : dict, optional
        Additional keyword arguments to pass to
        :class:`pyvista.Plotter`. Default is None.
    mesh_kws : dict, optional
        Additional keyword arguments to pass to
        :meth:`pyvista.Plotter.add_mesh`. Default is None.
    cbar_kws : dict, optional
        Additional keyword arguments to pass to
        :meth:`pyvista.Plotter.add_scalar_bar`. Default is None.
    silhouette_kws : dict, optional
        Additional keyword arguments to pass to
        :meth:`pyvista.Plotter.add_silhouette`. Only used when
        `show_silhouette=True`. Default is None.
    force_fetch : bool, optional
        If True, will re-download template data even if cached locally.
        Recommended to use periodically to refresh data. Default is False.
    data_dir : str or Path, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 0

    Notes
    -----
    **Available templates:**

    - 'aseg': FreeSurfer's automatic brain segmentation, includes major
      subcortical structures (thalamus, striatum, hippocampus, etc.)
      Generated from ``tpl-MNI152NLin2009cAsym_res-01_seg-aseg_dseg.nii.gz``
      from TemplateFlow. See `FreeSurferColorLUT
      <https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT>`_
      for region IDs.
    - 'tianS1-S4': Multi-level atlases from Tian et al. providing finer
      subdivisions of subcortical structures. Generated from
      ``Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S{1,2,3,4}_3T_2009cAsym.nii.gz``.
    - 'custom': User-provided atlas generated from volumetric data using
      :func:`_pv_make_subcortex_surfaces`

    **Template data updates:**

    Subcortical surface templates are periodically updated to add new atlases
    and optimize existing surface quality. To ensure you have the latest
    versions, it's recommended to set ``force_fetch=True`` occasionally to
    re-download and refresh your local cached data. This is especially important
    when:

    - New atlas templates are announced
    - Surface quality improvements are released
    - You encounter rendering issues with cached surfaces
    - Starting a new publication project

    **Data format:**

    The `parcel_data` dictionary maps region identifiers to scalar values:

    >>> parcel_data = {  # doctest: +SKIP
    ...     '10': 0.5,  # thalamus
    ...     '11': 0.7,  # caudate
    ...     '12': 0.6,  # putamen
    ... }

    Region identifiers depend on the selected template (integer IDs from
    FreeSurfer, Tian atlas, etc.).

    There is no intrinsic left/right distinction in subcortical atlases, so
    `include_keys` must specify which regions to plot for each hemisphere.

    For example, this is usually how you would set `include_keys`, when you want
    left and right structures plotted in their respective hemisphere panels.
    >>> include_keys = (['10', '11', '12'], ['49', '50', '51'])  # doctest: +SKIP

    In contrast, the following will display BOTH left and right structures
    in BOTH hemisphere panels:
    >>> include_keys = (['10', '11', '12', '49', '50', '51'])  # doctest: +SKIP

    **Layouts:**

    - 'default': Shows medial and lateral views for each hemisphere
    - 'single': Shows only one view (useful for custom camera angles)
    - 'row'/'column': Linear arrangements of all standard views

    **Parallel projection:**

    When comparing subcortical structures of very different sizes (e.g.,
    thalamus vs. amygdala), `parallel_projection=True` provides scale-invariant
    visualization, while `parallel_projection=False` uses perspective projection.

    **Lighting styles:**

    Different lighting presets affect surface appearance through ambient,
    diffuse, specular, and specular_power parameters:

    - Metallic: Low ambient (0.1), high specular (1.0)
    - Plastic: Balanced properties, moderate specular (0.3)
    - Shiny: High specular (0.8), high specular_power (50)

    **Jupyter notebooks:**

    When using in Jupyter, the function automatically sets `notebook=True` and
    `off_screen=True` for proper rendering.

    There can be various issues when plotting in Jupyter notebooks depending on
    your environment. For troubleshooting and detailed configuration options, see:

    - `Trame Jupyter Guide <https://kitware.github.io/trame/guide/jupyter/intro.html>`_
    - `PyVista Jupyter Documentation
      <https://docs.pyvista.org/user-guide/jupyter/>`_

    **Backend selection:**

    Choose the appropriate `jupyter_backend` for your use case:

    - `'trame'`: Best performance and interactivity (recommended)
    - `'html'`: Good interactivity, works in most environments
    - `'static'`: No interactivity but reliable fallback option

    If trame does not work in your environment, try html. The static option
    should always work as a last resort.

    **Customization with keyword arguments:**

    The `plotter_kws`, `mesh_kws`, `cbar_kws`, and `silhouette_kws` parameters
    allow flexible overriding of default settings. For example:

    - `plotter_kws={'window_size': (2000, 1500)}` for higher resolution
    - `mesh_kws={'ambient': 0.5}` to adjust material properties
    - `cbar_kws={'n_labels': 5}` for more colorbar labels
    - `silhouette_kws={'feature_angle': 30}` to adjust edge detection sensitivity

    Examples
    --------
    **Basic usage:**

    Plot random data on aseg template:

    >>> from netneurotools.plotting import pv_plot_subcortex  # doctest: +SKIP
    >>> parcel_data = {  # doctest: +SKIP
    ...     '10': 0.5, '11': 0.7, '12': 0.6, '13': 0.8,
    ...     '49': 0.4, '50': 0.6, '51': 0.5, '52': 0.7
    ... }
    >>> pl = pv_plot_subcortex(  # doctest: +SKIP
    ...     parcel_data,
    ...     template="aseg"
    ...     include_keys=(['10', '11', '12', '13'], ['49', '50', '51', '52'])
    ... )

    **Different atlases:**

    Use Tian atlas with finer subcortical subdivisions:

    >>> parcel_data_tian = {str(k): v for k, v in enumerate(  # doctest: +SKIP
    ...     np.random.random(16)
    ... )}
    >>> pl = pv_plot_subcortex(  # doctest: +SKIP
    ...     parcel_data_tian,
    ...     template="tianS2"  # Tian level 2 atlas
    ... )

    **Single hemisphere:**

    Plot only left hemisphere with custom include_keys:

    >>> include_keys = ['10', '11', '12']  # thalamus, caudate, putamen  # doctest: +SKIP
    >>> pl = pv_plot_subcortex(  # doctest: +SKIP
    ...     parcel_data,
    ...     template="aseg",
    ...     hemi="L",
    ...     include_keys=include_keys,
    ... )

    **Different layouts:**

    Compare all layout options:

    >>> for layout in ["default", "single", "row", "column"]:  # doctest: +SKIP
    ...     pl = pv_plot_subcortex(
    ...         parcel_data,
    ...         template="aseg",
    ...         layout=layout,
    ...         cbar_title=f"Layout: {layout}"
    ...     )

    **Custom colorbar and limits:**

    Set explicit color limits and colorbar title:

    >>> pl = pv_plot_subcortex(  # doctest: +SKIP
    ...     parcel_data,
    ...     template="aseg",
    ...     cmap="RdBu_r",
    ...     clim=(-1, 1),  # Symmetric limits
    ...     cbar_title="Activation (z-score)",
    ... )

    **Silhouette edges:**

    Add edge outlines for clarity, for example, this
    simulates the 2D flat drawing style:

    >>> pl = pv_plot_subcortex(  # doctest: +SKIP
    ...     parcel_data,
    ...     template="aseg",
    ...     lighting_style="none",
    ...     show_silhouette=True,
    ...     silhouette_kws={'color': 'black', 'line_width': 5},
    ... )

    **Saving figures:**

    Save as high-resolution PNG:

    >>> pl = pv_plot_subcortex(  # doctest: +SKIP
    ...     parcel_data,
    ...     template="aseg",
    ...     save_fig="subcortex_plot.png",
    ... )

    Save as vector graphics (SVG):

    >>> pl = pv_plot_subcortex(  # doctest: +SKIP
    ...     parcel_data,
    ...     template="aseg",
    ...     save_fig="subcortex_plot.svg",
    ... )

    **Lighting styles:**

    Explore different lighting presets:

    >>> for style in ["metallic", "plastic", "shiny", "glossy"]:  # doctest: +SKIP
    ...     pl = pv_plot_subcortex(
    ...         parcel_data,
    ...         template="aseg",
    ...         lighting_style=style,
    ...         cbar_title=f"Style: {style}",
    ...         save_fig=f"subcortex_{style}.png",
    ...     )

    **Advanced customization:**

    Combine multiple options for publication-quality figures:

    >>> pl = pv_plot_subcortex(  # doctest: +SKIP
    ...     parcel_data,
    ...     template="tianS2",
    ...     hemi="both",
    ...     layout="row",
    ...     cmap="plasma",
    ...     clim=(0, 1),
    ...     zoom_ratio=1.6,
    ...     show_colorbar=True,
    ...     cbar_title="Regional Connectivity",
    ...     lighting_style="shiny",
    ...     parallel_projection=True,
    ...     save_fig="publication_figure.png",
    ...     jupyter_backend=None,  # For script usage
    ... )

    **Custom surfaces:**

    Use user-defined subcortical surfaces generated from volumetric data:

    >>> from netneurotools.plotting import _pv_make_subcortex_surfaces  # doctest: +SKIP
    >>> # Assuming atlas.nii.gz contains custom volumetric segmentation
    >>> custom_surfs = _pv_make_subcortex_surfaces(  # doctest: +SKIP
    ...     "atlas.nii.gz",
    ...     include_keys=[1, 2, 3]
    ... )
    >>> pl = pv_plot_subcortex(  # doctest: +SKIP
    ...     parcel_data,
    ...     template="custom",
    ...     custom_surfaces=custom_surfs,
    ... )

    **Plotter customization for further manipulation:**

    Return plotter object without showing to add custom elements:

    >>> pl = pv_plot_subcortex(  # doctest: +SKIP
    ...     parcel_data,
    ...     template="aseg",
    ...     show_plot=False,  # Don't show yet
    ... )
    >>> # Add custom annotations
    >>> pl.add_text("Custom Title", position="upper_edge")  # doctest: +SKIP
    >>> pl.show()  # doctest: +SKIP
    """  # noqa: E501
    if not _has_pyvista:
        raise ImportError("PyVista is required for this function")

    if hemi not in ["L", "R", "both"]:
        raise ValueError(f"Unknown hemi: {hemi}")

    if include_keys is None:
        # If no specific regions selected, use all regions from parcel_data
        keys_list = list(parcel_data.keys())
        include_keys = (keys_list, keys_list) if hemi == "both" else keys_list

    if template in ["aseg", "tianS1", "tianS2", "tianS3", "tianS4"]:
        if hemi == "both":
            surf_pair = (
                _pv_load_subcortex_surfaces(
                    template, include_keys[0], force_fetch, data_dir, verbose),
                _pv_load_subcortex_surfaces(
                    template, include_keys[1], force_fetch, data_dir, verbose)
            )
        else:
            surf = _pv_load_subcortex_surfaces(
                template, include_keys, force_fetch, data_dir, verbose)

    elif template == "custom":
        if custom_surfaces is None:
            raise ValueError(
                "Must provide custom_surfaces for template='custom'. "
                "Recommended to provide a PyVista MultiBlock object or dict "
                "of {key: pyvista mesh} generated with _pv_make_subcortex_surfaces()"
            )
        if hemi == "both":
            surf_pair = (
                [custom_surfaces[k] for k in include_keys[0]],
                [custom_surfaces[k] for k in include_keys[1]],
            )
        else:
            surf = [custom_surfaces[k] for k in include_keys]
    else:
        raise ValueError(f"Unknown template: {template}")

    # Determine grid layout based on hemisphere and layout preference
    plotter_shape = _pv_get_plotter_shape(hemi, layout)

    # Extract parcel values and compute color scale
    # Determine color limits: use provided clim or calculate from data percentiles
    if hemi == "both":
        surf_data_pair = (
            np.array([parcel_data[str(k)] for k in include_keys[0]]),
            np.array([parcel_data[str(k)] for k in include_keys[1]])
        )
        _values = np.r_[surf_data_pair[0], surf_data_pair[1]]
    else:
        surf_data = np.array([parcel_data[str(k)] for k in include_keys])
        _values = surf_data

    if clim is not None:
        _vmin, _vmax = clim
    else:
        _vmin, _vmax = np.nanpercentile(_values, [2.5, 97.5])

    cmap = mpl.colormaps[cmap]
    cnorm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

    # Map normalized data values to RGBA colors and extract RGB components
    def _apply_colormap(arr):
        return [cmap(cnorm(val))[:3] for val in arr]

    if hemi == "both":
        surf_colors_pair = (
            _apply_colormap(surf_data_pair[0]),
            _apply_colormap(surf_data_pair[1]),
        )
    else:
        surf_colors = _apply_colormap(surf_data)

    plotter_settings, mesh_settings, cbar_settings, silhouette_settings = \
        _pv_update_settings(
            panel_size=panel_size,
            plotter_shape=plotter_shape,
            scalars=None,
            cmap=None,
            _vmin=_vmin,
            _vmax=_vmax,
            cbar_title=cbar_title,
            lighting_style=lighting_style,
            jupyter_backend=jupyter_backend,
            plotter_kws=plotter_kws,
            mesh_kws=mesh_kws,
            cbar_kws=cbar_kws,
            silhouette_kws=silhouette_kws
        )

    pl = pv.Plotter(shape=plotter_shape, **plotter_settings)

    if layout == "single":  # Single panel view
        if hemi == "both":
            _surf = surf_pair[0]
            _color = surf_colors_pair[0]
            _view_flip = True
        else:
            _surf = surf
            _color = surf_colors
            if hemi == "L":
                _view_flip = True
            else:
                _view_flip = False
        pl.subplot(0, 0)
        for _s, _c in zip(_surf, _color):
            pl.add_mesh(_s, color=_c, **mesh_settings)
            if show_silhouette:
                pl.add_silhouette(_s, **silhouette_settings)
        pl.view_yz(negative=_view_flip)
        pl.zoom_camera(zoom_ratio)

        if parallel_projection:
            pl.enable_parallel_projection()
    else:  # Multi-panel layout with multiple views
        if hemi == "both":  # Display 4 panels: 2 views per hemisphere
            if layout == "default":
                _pos = [(0, 0), (1, 0), (1, 1), (0, 1)]
            elif layout == "row":
                _pos = [(0, 0), (0, 1), (0, 2), (0, 3)]
            elif layout == "column":
                _pos = [(0, 0), (1, 0), (2, 0), (3, 0)]
            else:
                raise ValueError(f"Unknown layout: {layout}")

            _surf_list = [
                surf_pair[0], surf_pair[0], surf_pair[1], surf_pair[1]
            ]
            _color_list = [
                surf_colors_pair[0],
                surf_colors_pair[0],
                surf_colors_pair[1],
                surf_colors_pair[1]
            ]
            _view_flip_list = [True, False, True, False]

            for _xy, _surf, _color, _view_flip in zip(
                _pos, _surf_list, _color_list, _view_flip_list
            ):
                pl.subplot(*_xy)
                for _s, _c in zip(_surf, _color):
                    pl.add_mesh(_s, color=_c, **mesh_settings)
                    if show_silhouette:
                        pl.add_silhouette(_s, **silhouette_settings)
                pl.view_yz(negative=_view_flip)
                pl.zoom_camera(zoom_ratio)
                if parallel_projection:
                    pl.enable_parallel_projection()
        else:  # Display 2 panels: medial and lateral views of single hemisphere
            if layout == "default":
                _pos = [(0, 0), (0, 1)]
            elif layout == "row":
                _pos = [(0, 0), (0, 1)]
            elif layout == "column":
                _pos = [(0, 0), (1, 0)]
            else:
                raise ValueError(f"Unknown layout: {layout}")

            _surf_list = [surf, surf]
            _color_list = [surf_colors, surf_colors]
            if hemi == "L":
                _view_flip_list = [True, False]
            else:
                _view_flip_list = [False, True]

            for _xy, _surf, _color, _view_flip in zip(
                _pos, _surf_list, _color_list, _view_flip_list
            ):
                pl.subplot(*_xy)
                for _s, _c in zip(_surf, _color):
                    pl.add_mesh(_s, color=_c, **mesh_settings)
                    if show_silhouette:
                        pl.add_silhouette(_s, **silhouette_settings)
                pl.view_yz(negative=_view_flip)
                pl.zoom_camera(zoom_ratio)
                if parallel_projection:
                    pl.enable_parallel_projection()

    if show_colorbar:
        _pv_add_colorbar(
            pl=pl,
            layout=layout,
            _vmin=_vmin,
            _vmax=_vmax,
            cmap=cmap,
            cbar_settings=cbar_settings
        )

    if show_plot:
        if jupyter_backend is not None:
            pl.show(jupyter_backend=jupyter_backend)
        else:
            pl.show()

    if save_fig is not None:
        _pv_save_fig(pl, save_fig)

    return pl
