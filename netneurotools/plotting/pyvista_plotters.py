"""Functions for pyvista-based plotting."""

from pathlib import Path
import numpy as np
import nibabel as nib
import pyvista as pv
from netneurotools.datasets import (
    fetch_fslr_curated,
    fetch_fsaverage_curated,
    fetch_civet_curated,
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


def _pv_make_surface(template, surf="inflated", hemi=None, data_dir=None, verbose=0):
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
        ret_L = data[0].copy()
        ret_R = data[1].copy()
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


def pv_plot_surface(
    vertex_data,
    template,
    surf="inflated",
    hemi="both",
    layout="default",
    mask_medial=True,
    cmap="viridis",
    clim=None,
    zoom_ratio=1.0,
    show_colorbar=True,
    cbar_title=None,
    show_plot=True,
    jupyter_backend="html",
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

    Parameters
    ----------
    vertex_data : array-like or tuple of array-like
        Data array(s) to be plotted on the surface. If `hemi` is "both", this
        should be a tuple of two arrays. Otherwise, a single array.
    template : str
        Template to use for plotting. Options include 'fsaverage', 'fsaverage6',
        'fsaverage5', 'fsaverage4', 'fslr4k', 'fslr8k', 'fslr32k', 'fslr164k',
        'civet41k', 'civet164k'.
    surf : str, optional
        Surface to plot. Default is 'inflated'.
    hemi : str, optional
        Hemisphere to plot. Options include 'L', 'R', 'both'. Default is 'both'.
    layout : str, optional
        Layout of the plot. Options include 'default', 'single', 'row', 'column'.
        Default is 'default'.
    mask_medial : bool, optional
        Mask medial wall. Default is True.
    cmap : str, optional
        Colormap to use. Default is 'viridis'.
    clim : tuple, optional
        Colorbar limits. If None, will be set to 2.5th and 97.5th percentiles.
        Default is None.
    zoom_ratio : float, optional
        Zoom ratio for the camera. Default is 1.0.
    show_colorbar : bool, optional
        Whether to show the colorbar. Default is True.
    cbar_title : str, optional
        Title for the colorbar. Default is None.
    show_plot : bool, optional
        Whether to show the plot. Default is True.
    jupyter_backend : str, optional
        Jupyter backend to use. See `PyVista documentation
        <https://docs.pyvista.org/user-guide/jupyter/index.html#pyvista.set_jupyter_backend>`_
        for more details. Default is 'html'.
    lighting_style : str, optional
        Lighting style to use. Options include 'default', 'lightkit', 'threelights',
        'silhouette', 'metallic', 'plastic', 'shiny', 'glossy', 'ambient', 'plain'.
        Default is 'default'.
    save_fig : str or Path, optional
        Path (include file name) to save the figure. Default is None.

    Returns
    -------
    pl : PyVista.Plotter
        PyVista plotter object.

    Other Parameters
    ----------------
    plotter_kws : dict, optional
        Additional keyword arguments to pass to the `PyVista plotter
        <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter>`_.
        Default is None.
    mesh_kws : dict, optional
        Additional keyword arguments to pass to the `PyVista mesh
        <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh>`_.
        Default is None.
    cbar_kws : dict, optional
        Additional keyword arguments to pass to the `PyVista colorbar
        <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_scalar_bar>`_.
        Default is None.
    silhouette_kws : dict, optional
        Additional keyword arguments to pass to the `PyVista silhouette
        <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_silhouette>`_.
        Default is None.
    data_dir : str or Path, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 0
    """
    # setup data
    # could be a single array or a tuple of two arrays
    if hemi == "both":  # both hemispheres
        surf_pair = _pv_make_surface(
            template=template, surf=surf, data_dir=data_dir, verbose=verbose
        )
        if len(vertex_data) == 2:  # tuple or list of two arrays
            # check if data length matches number of vertices
            if not all(len(vertex_data[i]) == surf_pair[i].n_points for i in range(2)):
                raise ValueError("Data length mismatch")
        else:  # combined array
            # check if data length matches number of vertices
            if len(vertex_data) != surf_pair[0].n_points + surf_pair[1].n_points:
                raise ValueError("Data length mismatch")
            # convert long array to tuple
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
    elif hemi in ["L", "R"]:
        # single hemisphere
        surf = _pv_make_surface(
            template=template, surf=surf, hemi=hemi, data_dir=data_dir, verbose=verbose
        )
        if len(vertex_data) != surf.n_points:
            raise ValueError("Data length mismatch")

        if mask_medial:
            vertex_data = _mask_medial_wall(
                vertex_data, template, hemi=hemi, data_dir=data_dir, verbose=verbose
            )
        surf.point_data["vertex_data"] = vertex_data
    else:
        raise ValueError(f"Unknown hemi: {hemi}")

    # setup plotter shape based on layout
    if layout == "default":
        if hemi == "both":
            plotter_shape = (2, 2)
        else:
            plotter_shape = (1, 2)
    elif layout == "single":
        plotter_shape = (1, 1)
    elif layout == "row":
        if hemi == "both":
            plotter_shape = (1, 4)
        else:
            plotter_shape = (2, 1)
    elif layout == "column":
        if hemi == "both":
            plotter_shape = (4, 1)
        else:
            plotter_shape = (1, 2)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # setup color limits
    if clim is not None:
        _vmin, _vmax = clim
    else:
        if len(vertex_data) == 2:
            _values = np.c_[vertex_data[0], vertex_data[1]]
        else:
            _values = vertex_data
        _vmin, _vmax = np.nanpercentile(_values, [2.5, 97.5])

    # default plotter settings
    plotter_settings = dict(
        window_size=(350 * plotter_shape[1], 250 * plotter_shape[0]),
        border=False,
        lighting="three lights",
    )
    # notebook plotting
    if jupyter_backend is not None:
        plotter_settings.update(dict(notebook=True, off_screen=True))

    # default mesh settings
    mesh_settings = dict(
        scalars="vertex_data",
        smooth_shading=True,
        cmap=cmap,
        clim=(_vmin, _vmax),
        show_scalar_bar=False,
    )

    # lighting styles
    lighting_style_keys = ["ambient", "diffuse", "specular", "specular_power"]
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
    elif lighting_style == "silhouette":
        mesh_settings["lighting"] = "light kit"
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
    else:
        raise ValueError(f"Unknown lighting style: {lighting_style}")

    # default colorbar settings
    cbar_settings = dict(
        title=cbar_title,
        n_labels=2,
        label_font_size=10,
        title_font_size=12,
        font_family="arial",
        height=0.15,
    )

    # default silhouette settings
    silhouette_settings = dict(
        color="white",
        feature_angle=40
    )

    # update if provided with custom settings
    if plotter_kws is not None:
        plotter_settings.update(plotter_kws)
    if mesh_kws is not None:
        mesh_settings.update(mesh_kws)
    if cbar_kws is not None:
        cbar_settings.update(cbar_kws)
    if silhouette_kws is not None:
        silhouette_settings.update(silhouette_kws)

    pl = pv.Plotter(shape=plotter_shape, **plotter_settings)

    if layout == "single":  # single panel (1, 1)
        if hemi == "both":
            _surf = surf_pair[0].rotate_z(180)
        pl.subplot(0, 0)
        pl.add_mesh(_surf, **mesh_settings)
        pl.camera_position = "yz"
        pl.zoom_camera(zoom_ratio)
        if lighting_style == "silhouette":
            pl.add_silhouette(_surf, **silhouette_settings)
    else:  # multiple panels
        if hemi == "both":  # both hemi, 4 panels
            if layout == "default":
                _pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
            elif layout == "row":
                _pos = [(0, 0), (0, 2), (0, 1), (0, 3)]
            elif layout == "column":
                _pos = [(0, 0), (2, 0), (1, 0), (3, 0)]
            else:
                raise ValueError(f"Unknown layout: {layout}")
            _surf_list = [
                surf_pair[0].rotate_z(180),
                surf_pair[1],
                surf_pair[0],
                surf_pair[1].rotate_z(180),
            ]
            for _xy, _surf in zip(_pos, _surf_list):
                pl.subplot(*_xy)
                pl.add_mesh(_surf, **mesh_settings)
                pl.camera_position = "yz"
                pl.zoom_camera(zoom_ratio)
                if lighting_style == "silhouette":
                    pl.add_silhouette(_surf, **silhouette_settings)
        else:  # single hemi, 2 panels
            if layout == "default":
                _pos = [(0, 0), (0, 1)]
            elif layout == "row":
                _pos = [(0, 0), (0, 1)]
            elif layout == "column":
                _pos = [(0, 0), (1, 0)]
            else:
                raise ValueError(f"Unknown layout: {layout}")

            if hemi == "L":
                _surf_list = [surf.rotate_z(180), surf]
            else:
                _surf_list = [surf, surf.rotate_z(180)]

            for _xy, _surf in zip(_pos, _surf_list):
                pl.subplot(*_xy)
                pl.add_mesh(_surf, **mesh_settings)
                pl.camera_position = "yz"
                pl.zoom_camera(zoom_ratio)
                if lighting_style == "silhouette":
                    pl.add_silhouette(_surf, **silhouette_settings)

    if show_colorbar:
        cbar = pl.add_scalar_bar(**cbar_settings)
        cbar.GetLabelTextProperty().SetItalic(True)

    # setting the headlight (by default applied to all scenes)
    if lighting_style in ["default", "silhouette"] + list(
        lighting_style_presets.keys()
    ):
        light = pv.Light(light_type="headlight", intensity=0.2)
        pl.add_light(light)

    if show_plot:
        if jupyter_backend is not None:
            pl.show(jupyter_backend=jupyter_backend)
        else:
            pl.show()

    if save_fig is not None:
        _fname = Path(save_fig)
        if _fname.suffix in [".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff"]:
            pl.screenshot(_fname, return_img=False)
        elif _fname.suffix in [".svg", ".eps", ".ps", ".pdf", ".tex"]:
            pl.save_graphic(_fname)
        else:
            raise ValueError(f"Unknown file format: {save_fig}")

    return pl
