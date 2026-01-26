"""Functions for fetching template data."""

import json

from sklearn.utils import Bunch

from .datasets_utils import (
    SURFACE,
    _get_reference_info,
    _check_freesurfer_subjid,
    fetch_file,
)


def fetch_fsaverage(
    version="fsaverage", use_local=False, force=False, data_dir=None, verbose=1
):
    """
    Download files for fsaverage FreeSurfer template.

    This dataset contains surface files for the fsaverage template including
    original, white matter, pial, inflated, and spherical surfaces for both
    left and right hemispheres.

    If you used this data, please cite 1_, 2_, 3_.

    Parameters
    ----------
    version : str, optional
        One of {'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5',
        'fsaverage6'}. Default: 'fsaverage'
    use_local : bool, optional
        If True, will attempt to use local FreeSurfer data. Default: False

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['orig', 'white', 'smoothwm', 'pial',
        'inflated', 'sphere'], where corresponding values are Surface
        namedtuples containing filepaths for the left (L) and right (R)
        hemisphere surface files.

    Other Parameters
    ----------------
    force : bool, optional
        If True, will overwrite existing dataset. Default: False
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    The returned surfaces represent different stages of cortical surface
    reconstruction and transformations:

    - **orig**: Original surface extracted from the brain volume, representing
      the initial estimate of the cortical boundary before topology correction.
    - **white**: White matter surface, representing the boundary between white
      matter and gray matter (inner cortical surface).
    - **smoothwm**: Smoothed white matter surface, created by applying
      smoothing to the white surface for improved visualization and analysis.
    - **pial**: Pial surface, representing the outer boundary of the cortex
      (gray matter/CSF interface). This is commonly used for cortical thickness
      calculations and surface-based registration.
    - **inflated**: Inflated surface, where sulci and gyri are smoothed to
      make visualization of the entire cortical surface easier while preserving
      topology. Useful for visualizing data across the cortex without occlusion
      by folding patterns.
    - **sphere**: Spherical surface, where the cortical surface is mapped to a
      sphere. This is essential for surface-based registration, inter-subject
      alignment, and applying parcellations.

    Each surface can be loaded with neuroimaging tools like nibabel and used
    for surface-based analyses, visualization, or spatial transformations.

    In a typical FreeSurfer installation, these template surfaces can be found
    in the subjects directory under ``$FREESURFER_HOME/subjects/`` (e.g.,
    ``$FREESURFER_HOME/subjects/fsaverage/surf/``). When ``use_local=True``,
    this function will attempt to locate and use these local files instead of
    downloading them.

    Example directory tree:

    ::

        ~/nnt-data/tpl-fsaverage
        ├── fsaverage
        │   ├── LICENSE
        │   └── surf
        │       ├── lh.curv
        │       ├── lh.inflated
        │       ├── lh.inflated_avg
        │       ├── lh.orig
        │       ├── lh.orig_avg
        │       ├── lh.pial
        │       ├── lh.pial_avg
        │       ├── lh.smoothwm
        │       ├── lh.sphere
        │       ├── lh.sphere.reg.avg
        │       ├── lh.white
        │       ├── lh.white_avg
        │       ├── rh.curv
        │       ├── rh.inflated
        │       ├── rh.inflated_avg
        │       ├── rh.orig
        │       ├── rh.orig_avg
        │       ├── rh.pial
        │       ├── rh.pial_avg
        │       ├── rh.smoothwm
        │       ├── rh.sphere
        │       ├── rh.sphere.reg.avg
        │       ├── rh.white
        │       └── rh.white_avg
        ├── fsaverage3
        │   ├── LICENSE
        │   └── surf
        │       ├── lh.curv
        │       ├── lh.inflated
        │       ├── lh.inflated_avg
        │       ├── lh.orig
        │       ├── lh.orig_avg
        │       ├── lh.pial
        │       ├── lh.pial_avg
        │       ├── lh.smoothwm
        │       ├── lh.sphere
        │       ├── lh.sphere.reg.avg
        │       ├── lh.white
        │       ├── lh.white_avg
        │       ├── rh.curv
        │       ├── rh.inflated
        │       ├── rh.inflated_avg
        │       ├── rh.orig
        │       ├── rh.orig_avg
        │       ├── rh.pial
        │       ├── rh.pial_avg
        │       ├── rh.smoothwm
        │       ├── rh.sphere
        │       ├── rh.sphere.reg.avg
        │       ├── rh.white
        │       └── rh.white_avg
        ├── fsaverage4
        │   ├── LICENSE
        │   └── surf
        │       ├── lh.curv
        │       ├── lh.inflated
        │       ├── lh.inflated_avg
        │       ├── lh.orig
        │       ├── lh.orig_avg
        │       ├── lh.pial
        │       ├── lh.pial_avg
        │       ├── lh.smoothwm
        │       ├── lh.sphere
        │       ├── lh.sphere.reg.avg
        │       ├── lh.white
        │       ├── lh.white_avg
        │       ├── rh.curv
        │       ├── rh.inflated
        │       ├── rh.inflated_avg
        │       ├── rh.orig
        │       ├── rh.orig_avg
        │       ├── rh.pial
        │       ├── rh.pial_avg
        │       ├── rh.smoothwm
        │       ├── rh.sphere
        │       ├── rh.sphere.reg.avg
        │       ├── rh.white
        │       └── rh.white_avg
        ├── fsaverage5
        │   ├── LICENSE
        │   └── surf
        │       ├── lh.curv
        │       ├── lh.inflated
        │       ├── lh.inflated_avg
        │       ├── lh.orig
        │       ├── lh.orig_avg
        │       ├── lh.pial
        │       ├── lh.pial_avg
        │       ├── lh.smoothwm
        │       ├── lh.sphere
        │       ├── lh.sphere.reg.avg
        │       ├── lh.white
        │       ├── lh.white_avg
        │       ├── rh.curv
        │       ├── rh.inflated
        │       ├── rh.inflated_avg
        │       ├── rh.orig
        │       ├── rh.orig_avg
        │       ├── rh.pial
        │       ├── rh.pial_avg
        │       ├── rh.smoothwm
        │       ├── rh.sphere
        │       ├── rh.sphere.reg.avg
        │       ├── rh.white
        │       └── rh.white_avg
        └── fsaverage6
            ├── LICENSE
            └── surf
                ├── lh.curv
                ├── lh.inflated
                ├── lh.inflated_avg
                ├── lh.orig
                ├── lh.orig_avg
                ├── lh.pial
                ├── lh.pial_avg
                ├── lh.smoothwm
                ├── lh.sphere
                ├── lh.sphere.reg.avg
                ├── lh.white
                ├── lh.white_avg
                ├── rh.curv
                ├── rh.inflated
                ├── rh.inflated_avg
                ├── rh.orig
                ├── rh.orig_avg
                ├── rh.pial
                ├── rh.pial_avg
                ├── rh.smoothwm
                ├── rh.sphere
                ├── rh.sphere.reg.avg
                ├── rh.white
                └── rh.white_avg

        10 directories, 125 files

    Examples
    --------
    Load the fsaverage template surfaces:

    >>> surfaces = fetch_fsaverage(version='fsaverage')  # doctest: +SKIP
    >>> surfaces.keys()  # doctest: +SKIP
    dict_keys(['orig', 'white', 'smoothwm', 'pial', 'inflated', 'sphere'])

    Access the pial surface paths for left and right hemispheres:

    >>> surfaces.pial  # doctest: +SKIP
    Surface(L=PosixPath('~/nnt-data/tpl-fsaverage/fsaverage/surf/lh.pial'),
            R=PosixPath('~/nnt-data/tpl-fsaverage/fsaverage/surf/rh.pial'))

    Load the left pial surface with nibabel to examine its structure:

    >>> import nibabel as nib  # doctest: +SKIP
    >>> pial_left = nib.freesurfer.read_geometry(surfaces.pial.L)  # doctest: +SKIP
    >>> vertices, faces = pial_left  # doctest: +SKIP
    >>> print(f"Vertices: {vertices.shape}, Faces: {faces.shape}")  # doctest: +SKIP
    Vertices: (163842, 3), Faces: (327680, 3)

    References
    ----------
    .. [1] Anders M Dale, Bruce Fischl, and Martin I Sereno. Cortical
        surface-based analysis: i. segmentation and surface reconstruction.
        Neuroimage, 9(2):179\u2013194, 1999.
    .. [2] Bruce Fischl, Martin I Sereno, and Anders M Dale. Cortical
        surface-based analysis: ii: inflation, flattening, and a surface-based
        coordinate system. Neuroimage, 9(2):195\u2013207, 1999.
    .. [3] Bruce Fischl, Martin I Sereno, Roger BH Tootell, and Anders M Dale.
        High-resolution intersubject averaging and a coordinate system for the
        cortical surface. Human brain mapping, 8(4):272\u2013284, 1999.
    """
    versions = ["fsaverage", "fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6"]
    if version not in versions:
        raise ValueError(
            f"The version of fsaverage requested {version} does not "
            f"exist. Must be one of {versions}"
        )

    dataset_name = "tpl-fsaverage"
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ["orig", "white", "smoothwm", "pial", "inflated", "sphere"]

    if use_local:
        try:
            data_dir = _check_freesurfer_subjid(version)[1]
            data = {
                k: SURFACE(
                    data_dir / f"{version}/surf/lh.{k}",
                    data_dir / f"{version}/surf/rh.{k}",
                )
                for k in keys
            }
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Local FreeSurfer data for {version} not found. "
                "Please ensure FreeSurfer is installed and properly set up."
            ) from None
    else:
        fetched = fetch_file(
            dataset_name, keys=version, force=force, data_dir=data_dir, verbose=verbose
        )

        data = {
            k: SURFACE(
                fetched / f"surf/lh.{k}",
                fetched / f"surf/rh.{k}",
            )
            for k in keys
        }

    return Bunch(**data)


def fetch_fsaverage_curated(version="fsaverage", force=False, data_dir=None, verbose=1):
    """
    Download files for fsaverage FreeSurfer template.

    This dataset contains surface geometry files (white, pial, inflated,
    sphere), medial wall labels, and surface shape files (sulcal depth and
    vertex area) in GIFTI format for the fsaverage template at various
    densities.

    If you used this data, please cite 1_, 2_, 3_, 4_.

    Parameters
    ----------
    version : str, optional
        One of {'fsaverage', 'fsaverage4', 'fsaverage5', 'fsaverage6'}.
        Default: 'fsaverage'

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['white', 'pial', 'inflated',
        'sphere', 'medial', 'sulc', 'vaavg'], where corresponding values are
        Surface namedtuples containing filepaths for the left (L) and right
        (R) hemisphere files in GIFTI format.

    Other Parameters
    ----------------
    force : bool, optional
        If True, will overwrite existing dataset. Default: False
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    This function fetches curated fsaverage surfaces from the neuromaps
    package (see `neuromaps.datasets.fetch_fsaverage <https://netneurolab.github.io/neuromaps/generated/neuromaps.datasets.fetch_fsaverage.html>`_).
    All files are provided in GIFTI format (.gii) rather than FreeSurfer's
    native format.

    The returned files include:

    - **white**: White matter surface geometry (.surf.gii), representing the
        boundary between white matter and gray matter. Corresponds to FreeSurfer
        surfaces 'lh.white' and 'rh.white'.
    - **pial**: Pial surface geometry (.surf.gii), representing the outer
        cortical boundary. Corresponds to FreeSurfer surfaces 'lh.pial' and
        'rh.pial'.
    - **inflated**: Inflated surface geometry (.surf.gii) for improved
        visualization of sulci and gyri. Corresponds to FreeSurfer surfaces
        'lh.inflated' and 'rh.inflated'.
    - **sphere**: Spherical surface geometry (.surf.gii) used for surface-based
        registration and applying parcellations. Corresponds to FreeSurfer
        surfaces 'lh.sphere' and 'rh.sphere'.
    - **medial**: Medial wall mask (.label.gii) indicating vertices to exclude
        from analyses (vertices with no cortex). Not a standard FreeSurfer
        output; derived by neuromaps to mark the no-medial-wall vertices.
    - **sulc**: Sulcal depth map (.shape.gii) providing sulcal/gyral patterns
        on the midthickness surface. Corresponds to FreeSurfer 'lh.sulc' and
        'rh.sulc' values resampled to the midthickness surface.
    - **vaavg**: Vertex area map (.shape.gii) representing the average vertex
        area on the midthickness surface. Not a standard FreeSurfer output;
        computed from mesh triangle areas and averaged per vertex.

    The vertex density varies by version: fsaverage (164k vertices),
    fsaverage6 (41k), fsaverage5 (10k), and fsaverage4 (3k).

    Example directory tree:

    ::

        ~/nnt-data/tpl-fsaverage_curated
        ├── fsaverage
        │   ├── tpl-fsaverage_den-164k_hemi-L_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsaverage_den-164k_hemi-L_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-164k_hemi-L_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-164k_hemi-L_inflated.surf.gii
        │   ├── tpl-fsaverage_den-164k_hemi-L_pial.surf.gii
        │   ├── tpl-fsaverage_den-164k_hemi-L_sphere.surf.gii
        │   ├── tpl-fsaverage_den-164k_hemi-L_white.surf.gii
        │   ├── tpl-fsaverage_den-164k_hemi-R_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsaverage_den-164k_hemi-R_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-164k_hemi-R_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-164k_hemi-R_inflated.surf.gii
        │   ├── tpl-fsaverage_den-164k_hemi-R_pial.surf.gii
        │   ├── tpl-fsaverage_den-164k_hemi-R_sphere.surf.gii
        │   └── tpl-fsaverage_den-164k_hemi-R_white.surf.gii
        ├── fsaverage4
        │   ├── tpl-fsaverage_den-3k_hemi-L_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsaverage_den-3k_hemi-L_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-3k_hemi-L_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-3k_hemi-L_inflated.surf.gii
        │   ├── tpl-fsaverage_den-3k_hemi-L_pial.surf.gii
        │   ├── tpl-fsaverage_den-3k_hemi-L_sphere.surf.gii
        │   ├── tpl-fsaverage_den-3k_hemi-L_white.surf.gii
        │   ├── tpl-fsaverage_den-3k_hemi-R_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsaverage_den-3k_hemi-R_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-3k_hemi-R_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-3k_hemi-R_inflated.surf.gii
        │   ├── tpl-fsaverage_den-3k_hemi-R_pial.surf.gii
        │   ├── tpl-fsaverage_den-3k_hemi-R_sphere.surf.gii
        │   └── tpl-fsaverage_den-3k_hemi-R_white.surf.gii
        ├── fsaverage5
        │   ├── tpl-fsaverage_den-10k_hemi-L_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsaverage_den-10k_hemi-L_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-10k_hemi-L_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-10k_hemi-L_inflated.surf.gii
        │   ├── tpl-fsaverage_den-10k_hemi-L_pial.surf.gii
        │   ├── tpl-fsaverage_den-10k_hemi-L_sphere.surf.gii
        │   ├── tpl-fsaverage_den-10k_hemi-L_white.surf.gii
        │   ├── tpl-fsaverage_den-10k_hemi-R_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsaverage_den-10k_hemi-R_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-10k_hemi-R_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsaverage_den-10k_hemi-R_inflated.surf.gii
        │   ├── tpl-fsaverage_den-10k_hemi-R_pial.surf.gii
        │   ├── tpl-fsaverage_den-10k_hemi-R_sphere.surf.gii
        │   └── tpl-fsaverage_den-10k_hemi-R_white.surf.gii
        └── fsaverage6
            ├── tpl-fsaverage_den-41k_hemi-L_desc-nomedialwall_dparc.label.gii
            ├── tpl-fsaverage_den-41k_hemi-L_desc-sulc_midthickness.shape.gii
            ├── tpl-fsaverage_den-41k_hemi-L_desc-vaavg_midthickness.shape.gii
            ├── tpl-fsaverage_den-41k_hemi-L_inflated.surf.gii
            ├── tpl-fsaverage_den-41k_hemi-L_pial.surf.gii
            ├── tpl-fsaverage_den-41k_hemi-L_sphere.surf.gii
            ├── tpl-fsaverage_den-41k_hemi-L_white.surf.gii
            ├── tpl-fsaverage_den-41k_hemi-R_desc-nomedialwall_dparc.label.gii
            ├── tpl-fsaverage_den-41k_hemi-R_desc-sulc_midthickness.shape.gii
            ├── tpl-fsaverage_den-41k_hemi-R_desc-vaavg_midthickness.shape.gii
            ├── tpl-fsaverage_den-41k_hemi-R_inflated.surf.gii
            ├── tpl-fsaverage_den-41k_hemi-R_pial.surf.gii
            ├── tpl-fsaverage_den-41k_hemi-R_sphere.surf.gii
            └── tpl-fsaverage_den-41k_hemi-R_white.surf.gii

        4 directories, 56 files


    Examples
    --------
    Load the fsaverage curated template surfaces:

    >>> surfaces = fetch_fsaverage_curated(version='fsaverage')  # doctest: +SKIP
    >>> surfaces.keys()  # doctest: +SKIP
    dict_keys(['white', 'pial', 'inflated', 'sphere', 'medial', 'sulc', 'vaavg'])

    Access the pial surface GIFTI files:

    >>> surfaces.pial  # doctest: +SKIP
    Surface(L=PosixPath('~/nnt-data/tpl-fsaverage_curated/fsaverage/tpl-fsaverage_den-164k_hemi-L_pial.surf.gii'),
            R=PosixPath('~/nnt-data/tpl-fsaverage_curated/fsaverage/tpl-fsaverage_den-164k_hemi-R_pial.surf.gii'))

    Load the left pial surface with nibabel:

    >>> import nibabel as nib  # doctest: +SKIP
    >>> pial_left = nib.load(surfaces.pial.L)  # doctest: +SKIP
    >>> vertices = pial_left.agg_data('pointset')  # doctest: +SKIP
    >>> faces = pial_left.agg_data('triangle')  # doctest: +SKIP
    >>> print(f"Vertices: {vertices.shape}, Faces: {faces.shape}")  # doctest: +SKIP
    Vertices: (163842, 3), Faces: (327680, 3)

    Load and examine the sulcal depth data:

    >>> sulc_left = nib.load(surfaces.sulc.L)  # doctest: +SKIP
    >>> sulc_data = sulc_left.agg_data()  # doctest: +SKIP
    >>> sulc_min, sulc_max = sulc_data.min(), sulc_data.max()  # doctest: +SKIP
    >>> print(f"Sulcal depth range: {sulc_min:.2f} to {sulc_max:.2f}")  # doctest: +SKIP
    Sulcal depth range: -1.78 to 1.88

    References
    ----------
    .. [1] Anders M Dale, Bruce Fischl, and Martin I Sereno. Cortical
        surface-based analysis: i. segmentation and surface reconstruction.
        Neuroimage, 9(2):179\u2013194, 1999.
    .. [2] Bruce Fischl, Martin I Sereno, and Anders M Dale. Cortical
        surface-based analysis: ii: inflation, flattening, and a surface-based
        coordinate system. Neuroimage, 9(2):195\u2013207, 1999.
    .. [3] Bruce Fischl, Martin I Sereno, Roger BH Tootell, and Anders M Dale.
        High-resolution intersubject averaging and a coordinate system for the
        cortical surface. Human brain mapping, 8(4):272\u2013284, 1999.
    .. [4] Ross D Markello, Justine Y Hansen, Zhen-Qi Liu, Vincent Bazinet,
        Golia Shafiei, Laura E Su\u00e1rez, Nadia Blostein, Jakob Seidlitz,
        Sylvain Baillet, Theodore D Satterthwaite, and others. Neuromaps:
        structural and functional interpretation of brain maps. Nature Methods,
        19(11):1472\u20131479, 2022.
    """
    versions = ["fsaverage", "fsaverage6", "fsaverage5", "fsaverage4"]
    if version not in versions:
        raise ValueError(
            f"The version of fsaverage requested {version} does not "
            f"exist. Must be one of {versions}"
        )

    dataset_name = "tpl-fsaverage_curated"
    _get_reference_info("tpl-fsaverage_curated", verbose=verbose)

    keys = ["white", "pial", "inflated", "sphere", "medial", "sulc", "vaavg"]
    keys_suffix = {
        "white": "white.surf",
        "pial": "pial.surf",
        "inflated": "inflated.surf",
        "sphere": "sphere.surf",
        "medial": "desc-nomedialwall_dparc.label",
        "sulc": "desc-sulc_midthickness.shape",
        "vaavg": "desc-vaavg_midthickness.shape",
    }
    version_density = {
        "fsaverage": "164k",
        "fsaverage6": "41k",
        "fsaverage5": "10k",
        "fsaverage4": "3k",
    }
    density = version_density[version]

    fetched = fetch_file(
        dataset_name, keys=version, force=force, data_dir=data_dir, verbose=verbose
    )

    # deal with default neuromaps directory structure in the archive
    if not fetched.exists():
        import shutil

        shutil.move(fetched.parent / "atlases/fsaverage", fetched)
        shutil.rmtree(fetched.parent / "atlases")

    data = {
        k: SURFACE(
            fetched / f"tpl-fsaverage_den-{density}_hemi-L_{keys_suffix[k]}.gii",
            fetched / f"tpl-fsaverage_den-{density}_hemi-R_{keys_suffix[k]}.gii",
        )
        for k in keys
    }

    return Bunch(**data)


def fetch_hcp_standards(force=False, data_dir=None, verbose=1):
    """
    Fetch HCP standard mesh atlases for converting between FreeSurfer and HCP.

    This dataset contains standard mesh atlases used by Connectome Workbench
    to convert and register data between FreeSurfer fsaverage space and HCP
    fsLR space. It includes spherical templates for fsaverage and fsLR at
    multiple vertex densities (e.g., 164k, 59k, 32k), mapping spheres between
    fs (hemisphere-specific) and fsLR, and midthickness vertex area averages
    (``va_avg``) for resampling and area-preserving operations.

    The original file was from 3_, but is no longer available. The archived
    file is available from 4_.

    If you used this data, please cite 1_, 2_.

    Returns
    -------
    standards : str
        Filepath to standard_mesh_atlases directory

    Other Parameters
    ----------------
    force : bool, optional
        If True, will overwrite existing dataset. Default: False
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    Returns the path to the `standard_mesh_atlases` directory containing
    curated GIFTI files used for conversions between FreeSurfer fsaverage and
    HCP fsLR spaces, including spherical templates and midthickness vertex-area
    maps at multiple densities.

    Example directory tree:

    ::

        ~/nnt-data/tpl-hcp_standards/standard_mesh_atlases
        ├── fsaverage.L_LR.spherical_std.164k_fs_LR.surf.gii
        ├── fsaverage.R_LR.spherical_std.164k_fs_LR.surf.gii
        ├── fs_L
        │   ├── fsaverage.L.sphere.164k_fs_L.surf.gii
        │   └── fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.surf.gii
        ├── fs_R
        │   ├── fsaverage.R.sphere.164k_fs_R.surf.gii
        │   └── fs_R-to-fs_LR_fsaverage.R_LR.spherical_std.164k_fs_R.surf.gii
        ├── L.sphere.32k_fs_LR.surf.gii
        ├── L.sphere.59k_fs_LR.surf.gii
        ├── resample_fsaverage
        │   ├── fsaverage4.L.midthickness_va_avg.3k_fsavg_L.shape.gii
        │   ├── fsaverage4.R.midthickness_va_avg.3k_fsavg_R.shape.gii
        │   ├── fsaverage4_std_sphere.L.3k_fsavg_L.surf.gii
        │   ├── fsaverage4_std_sphere.R.3k_fsavg_R.surf.gii
        │   ├── fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii
        │   ├── fsaverage5.R.midthickness_va_avg.10k_fsavg_R.shape.gii
        │   ├── fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii
        │   ├── fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii
        │   ├── fsaverage6.L.midthickness_va_avg.41k_fsavg_L.shape.gii
        │   ├── fsaverage6.R.midthickness_va_avg.41k_fsavg_R.shape.gii
        │   ├── fsaverage6_std_sphere.L.41k_fsavg_L.surf.gii
        │   ├── fsaverage6_std_sphere.R.41k_fsavg_R.surf.gii
        │   ├── fsaverage.L.midthickness_va_avg.164k_fsavg_L.shape.gii
        │   ├── fsaverage.R.midthickness_va_avg.164k_fsavg_R.shape.gii
        │   ├── fsaverage_std_sphere.L.164k_fsavg_L.surf.gii
        │   ├── fsaverage_std_sphere.R.164k_fsavg_R.surf.gii
        │   ├── fs_LR-deformed_to-fsaverage.L.sphere.164k_fs_LR.surf.gii
        │   ├── fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii
        │   ├── fs_LR-deformed_to-fsaverage.L.sphere.59k_fs_LR.surf.gii
        │   ├── fs_LR-deformed_to-fsaverage.R.sphere.164k_fs_LR.surf.gii
        │   ├── fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii
        │   ├── fs_LR-deformed_to-fsaverage.R.sphere.59k_fs_LR.surf.gii
        │   ├── fs_LR.L.midthickness_va_avg.164k_fs_LR.shape.gii
        │   ├── fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii
        │   ├── fs_LR.L.midthickness_va_avg.59k_fs_LR.shape.gii
        │   ├── fs_LR.R.midthickness_va_avg.164k_fs_LR.shape.gii
        │   ├── fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii
        │   └── fs_LR.R.midthickness_va_avg.59k_fs_LR.shape.gii
        ├── R.sphere.32k_fs_LR.surf.gii
        └── R.sphere.59k_fs_LR.surf.gii

        3 directories, 38 files

    Examples
    --------
    Load the standards directory and inspect contents:

    >>> standards = fetch_hcp_standards()  # doctest: +SKIP
    >>> print(standards)  # doctest: +SKIP
    PosixPath('~/nnt-data/tpl-hcp_standards/standard_mesh_atlases')

    List the fsLR 32k spherical templates:

    >>> import pathlib  # doctest: +SKIP
    >>> list((standards).glob('L.sphere.32k_fs_LR.surf.gii'))  # doctest: +SKIP
    [PosixPath('~/nnt-data/tpl-hcp_standards/standard_mesh_atlases/L.sphere.32k_fs_LR.surf.gii')]

    Load a sphere surface with nibabel and examine geometry:

    >>> import nibabel as nib  # doctest: +SKIP
    >>> gii = nib.load(standards / 'L.sphere.32k_fs_LR.surf.gii')  # doctest: +SKIP
    >>> vertices = gii.agg_data('pointset')  # doctest: +SKIP
    >>> faces = gii.agg_data('triangle')  # doctest: +SKIP
    >>> vertices.shape, faces.shape  # doctest: +SKIP
    ((32492, 3), (64980, 3))

    References
    ----------
    .. [1] David C Van Essen, Kamil Ugurbil, Edward Auerbach, Deanna
        Barch,Timothy EJ Behrens, Richard Bucholz, Acer Chang, Liyong Chen,
        Maurizio Corbetta, Sandra W Curtiss, and others. The human connectome
        project: a data acquisition perspective. Neuroimage,
        62(4):2222\u20132231, 2012.
    .. [2] Matthew F Glasser, Stamatios N Sotiropoulos, J Anthony Wilson,
        Timothy S Coalson, Bruce Fischl, Jesper L Andersson, Junqian Xu, Saad
        Jbabdi, Matthew Webster, Jonathan R Polimeni, and others. The minimal
        preprocessing pipelines for the human connectome project. Neuroimage,
        80:105\u2013124, 2013.
    .. [3] http://brainvis.wustl.edu/workbench/standard_mesh_atlases.zip
    .. [4] https://web.archive.org/web/20220121035833/http://brainvis.wustl.edu/workbench/standard_mesh_atlases.zip
    """
    dataset_name = "tpl-hcp_standards"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(
        dataset_name,
        keys="standard_mesh_atlases",
        force=force,
        data_dir=data_dir,
        verbose=verbose,
    )

    return fetched


def fetch_fslr_curated(version="fslr32k", force=False, data_dir=None, verbose=1):
    """
    Download files for HCP fsLR template.

    This dataset contains surface geometry files (midthickness, inflated,
    veryinflated [where available], sphere), medial wall labels, and surface
    shape files (sulcal depth and vertex area) in GIFTI format for the HCP fsLR
    template at various densities.

    If you used this data, please cite 1_, 2_, 3_.

    Parameters
    ----------
    version : str, optional
        One of {"fslr4k", "fslr8k", "fslr32k", "fslr164k"}. Default: 'fslr32k'

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['midthickness', 'inflated',
        'veryinflated' (except for 'fslr4k'/'fslr8k'), 'sphere', 'medial',
        'sulc', 'vaavg'], where corresponding values are Surface namedtuples
        containing filepaths for the left (L) and right (R) hemisphere files
        in GIFTI format.

    Other Parameters
    ----------------
    force : bool, optional
        If True, will overwrite existing dataset. Default: False
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    This function fetches curated fsLR surfaces from the neuromaps
    package (see `neuromaps.datasets.fetch_fslr <https://netneurolab.github.io/neuromaps/generated/neuromaps.datasets.fetch_fslr.html>`_).
    All files are provided in GIFTI format (.gii). The fsLR template is the
    HCP standard mesh used for group analyses and cross-subject alignment.


    The returned files include:

    - **midthickness**: Midthickness surface geometry (.surf.gii), halfway
        between white and pial surfaces; often preferred for data mapping.
    - **inflated**: Inflated surface geometry (.surf.gii) for improved
        visualization of sulci and gyri.
    - **veryinflated**: Very inflated surface geometry (.surf.gii) providing
        additional smoothing; not available for 'fslr4k'/'fslr8k'.
    - **sphere**: Spherical surface geometry (.surf.gii) used for surface-based
        registration and applying parcellations.
    - **medial**: Medial wall mask (.label.gii) indicating vertices to exclude
        from analyses (vertices with no cortex).
    - **sulc**: Sulcal depth map (.shape.gii) providing sulcal/gyral patterns
        on the midthickness surface.
    - **vaavg**: Vertex area map (.shape.gii) representing the average vertex
        area on the midthickness surface.

    The vertex density varies by version: fslr4k (≈4k vertices), fslr8k (≈8k),
    fslr32k (≈32k), and fslr164k (≈164k) per hemisphere.

    Example directory tree:

    ::

        ~/nnt-data/tpl-fslr_curated
        ├── fslr164k
        │   ├── README.md
        │   ├── tpl-fsLR_den-164k_hemi-L_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsLR_den-164k_hemi-L_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsLR_den-164k_hemi-L_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsLR_den-164k_hemi-L_inflated.surf.gii
        │   ├── tpl-fsLR_den-164k_hemi-L_midthickness.surf.gii
        │   ├── tpl-fsLR_den-164k_hemi-L_sphere.surf.gii
        │   ├── tpl-fsLR_den-164k_hemi-L_veryinflated.surf.gii
        │   ├── tpl-fsLR_den-164k_hemi-R_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsLR_den-164k_hemi-R_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsLR_den-164k_hemi-R_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsLR_den-164k_hemi-R_inflated.surf.gii
        │   ├── tpl-fsLR_den-164k_hemi-R_midthickness.surf.gii
        │   ├── tpl-fsLR_den-164k_hemi-R_sphere.surf.gii
        │   ├── tpl-fsLR_den-164k_hemi-R_veryinflated.surf.gii
        │   ├── tpl-fsLR_space-fsaverage_den-164k_hemi-L_sphere.surf.gii
        │   └── tpl-fsLR_space-fsaverage_den-164k_hemi-R_sphere.surf.gii
        ├── fslr32k
        │   ├── README.md
        │   ├── tpl-fsLR_den-32k_hemi-L_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsLR_den-32k_hemi-L_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsLR_den-32k_hemi-L_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsLR_den-32k_hemi-L_inflated.surf.gii
        │   ├── tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii
        │   ├── tpl-fsLR_den-32k_hemi-L_sphere.surf.gii
        │   ├── tpl-fsLR_den-32k_hemi-L_veryinflated.surf.gii
        │   ├── tpl-fsLR_den-32k_hemi-R_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsLR_den-32k_hemi-R_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsLR_den-32k_hemi-R_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsLR_den-32k_hemi-R_inflated.surf.gii
        │   ├── tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii
        │   ├── tpl-fsLR_den-32k_hemi-R_sphere.surf.gii
        │   ├── tpl-fsLR_den-32k_hemi-R_veryinflated.surf.gii
        │   ├── tpl-fsLR_space-fsaverage_den-32k_hemi-L_sphere.surf.gii
        │   └── tpl-fsLR_space-fsaverage_den-32k_hemi-R_sphere.surf.gii
        ├── fslr4k
        │   ├── tpl-fsLR_den-4k_hemi-L_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsLR_den-4k_hemi-L_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsLR_den-4k_hemi-L_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsLR_den-4k_hemi-L_inflated.surf.gii
        │   ├── tpl-fsLR_den-4k_hemi-L_midthickness.surf.gii
        │   ├── tpl-fsLR_den-4k_hemi-L_sphere.surf.gii
        │   ├── tpl-fsLR_den-4k_hemi-R_desc-nomedialwall_dparc.label.gii
        │   ├── tpl-fsLR_den-4k_hemi-R_desc-sulc_midthickness.shape.gii
        │   ├── tpl-fsLR_den-4k_hemi-R_desc-vaavg_midthickness.shape.gii
        │   ├── tpl-fsLR_den-4k_hemi-R_inflated.surf.gii
        │   ├── tpl-fsLR_den-4k_hemi-R_midthickness.surf.gii
        │   ├── tpl-fsLR_den-4k_hemi-R_sphere.surf.gii
        │   ├── tpl-fsLR_space-fsaverage_den-4k_hemi-L_sphere.surf.gii
        │   └── tpl-fsLR_space-fsaverage_den-4k_hemi-R_sphere.surf.gii
        └── fslr8k
            ├── tpl-fsLR_den-8k_hemi-L_desc-nomedialwall_dparc.label.gii
            ├── tpl-fsLR_den-8k_hemi-L_desc-sulc_midthickness.shape.gii
            ├── tpl-fsLR_den-8k_hemi-L_desc-vaavg_midthickness.shape.gii
            ├── tpl-fsLR_den-8k_hemi-L_inflated.surf.gii
            ├── tpl-fsLR_den-8k_hemi-L_midthickness.surf.gii
            ├── tpl-fsLR_den-8k_hemi-L_sphere.surf.gii
            ├── tpl-fsLR_den-8k_hemi-R_desc-nomedialwall_dparc.label.gii
            ├── tpl-fsLR_den-8k_hemi-R_desc-sulc_midthickness.shape.gii
            ├── tpl-fsLR_den-8k_hemi-R_desc-vaavg_midthickness.shape.gii
            ├── tpl-fsLR_den-8k_hemi-R_inflated.surf.gii
            ├── tpl-fsLR_den-8k_hemi-R_midthickness.surf.gii
            ├── tpl-fsLR_den-8k_hemi-R_sphere.surf.gii
            ├── tpl-fsLR_space-fsaverage_den-8k_hemi-L_sphere.surf.gii
            └── tpl-fsLR_space-fsaverage_den-8k_hemi-R_sphere.surf.gii

        4 directories, 62 files

    Examples
    --------
    Load the fsLR curated template surfaces:

    >>> surfaces = fetch_fslr_curated(version='fslr32k')  # doctest: +SKIP
    >>> surfaces.keys()  # doctest: +SKIP
    dict_keys(['midthickness', 'inflated', 'veryinflated', 'sphere', 'medial',
               'sulc', 'vaavg'])

    Access the midthickness surface GIFTI files:

    >>> surfaces.midthickness  # doctest: +SKIP
    Surface(L=PosixPath('~/nnt-data/tpl-fslr_curated/fslr32k/tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii'),
            R=PosixPath('~/nnt-data/tpl-fslr_curated/fslr32k/tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii'))

    Load the left midthickness surface with nibabel:

    >>> import nibabel as nib  # doctest: +SKIP
    >>> gii = nib.load(surfaces.midthickness.L)  # doctest: +SKIP
    >>> vertices = gii.agg_data('pointset')  # doctest: +SKIP
    >>> faces = gii.agg_data('triangle')  # doctest: +SKIP
    >>> print(vertices.shape, faces.shape)  # doctest: +SKIP
    (32492, 3) (64980, 3)

    Load and examine the sulcal depth data:

    >>> sulc_left = nib.load(surfaces.sulc.L)  # doctest: +SKIP
    >>> sulc_data = sulc_left.agg_data()  # doctest: +SKIP
    >>> float(sulc_data.min()), float(sulc_data.max())  # doctest: +SKIP
    (-1.6234848499298096, 1.1611071825027466)

    References
    ----------
    .. [1] David C Van Essen, Kamil Ugurbil, Edward Auerbach, Deanna
        Barch,Timothy EJ Behrens, Richard Bucholz, Acer Chang, Liyong Chen,
        Maurizio Corbetta, Sandra W Curtiss, and others. The human connectome
        project: a data acquisition perspective. Neuroimage,
        62(4):2222\u20132231, 2012.
    .. [2] Matthew F Glasser, Stamatios N Sotiropoulos, J Anthony Wilson,
        Timothy S Coalson, Bruce Fischl, Jesper L Andersson, Junqian Xu, Saad
        Jbabdi, Matthew Webster, Jonathan R Polimeni, and others. The minimal
        preprocessing pipelines for the human connectome project. Neuroimage,
        80:105\u2013124, 2013.
    .. [3] Ross D Markello, Justine Y Hansen, Zhen-Qi Liu, Vincent Bazinet,
        Golia Shafiei, Laura E Su\u00e1rez, Nadia Blostein, Jakob Seidlitz,
        Sylvain Baillet, Theodore D Satterthwaite, and others. Neuromaps:
        structural and functional interpretation of brain maps. Nature Methods,
        19(11):1472\u20131479, 2022.
    """
    versions = ["fslr4k", "fslr8k", "fslr32k", "fslr164k"]
    if version not in versions:
        raise ValueError(
            f"The version of fsaverage requested {version} does not "
            f"exist. Must be one of {versions}"
        )

    dataset_name = "tpl-fslr_curated"
    _get_reference_info("tpl-fslr_curated", verbose=verbose)

    keys = [
        "midthickness",
        "inflated",
        "veryinflated",
        "sphere",
        "medial",
        "sulc",
        "vaavg",
    ]
    if version in ["fslr4k", "fslr8k"]:
        keys.remove("veryinflated")
    keys_suffix = {
        "midthickness": "midthickness.surf",
        "inflated": "inflated.surf",
        "veryinflated": "veryinflated.surf",
        "sphere": "sphere.surf",
        "medial": "desc-nomedialwall_dparc.label",
        "sulc": "desc-sulc_midthickness.shape",
        "vaavg": "desc-vaavg_midthickness.shape",
    }
    version_density = {
        "fslr4k": "4k",
        "fslr8k": "8k",
        "fslr32k": "32k",
        "fslr164k": "164k",
    }
    density = version_density[version]

    fetched = fetch_file(
        dataset_name, keys=version, force=force, data_dir=data_dir, verbose=verbose
    )

    # deal with default neuromaps directory structure in the archive
    if not fetched.exists():
        import shutil

        shutil.move(fetched.parent / "atlases/fsLR", fetched)
        shutil.rmtree(fetched.parent / "atlases")

    data = {
        k: SURFACE(
            fetched / f"tpl-fsLR_den-{density}_hemi-L_{keys_suffix[k]}.gii",
            fetched / f"tpl-fsLR_den-{density}_hemi-R_{keys_suffix[k]}.gii",
        )
        for k in keys
    }

    return Bunch(**data)


def fetch_civet(density="41k", version="v1", force=False, data_dir=None, verbose=1):
    """
    Fetch CIVET surface files.

    This dataset contains midthickness and white matter surface files for the
    CIVET template in OBJ format, registered to ICBM152 space. CIVET is a
    fully automated structural image processing pipeline developed at the
    Montreal Neurological Institute.

    If you used this data, please cite 1_, 2_, 3_.

    Parameters
    ----------
    density : {'41k', '164k'}, optional
        Which density of the CIVET-space geometry files to fetch. The
        high-resolution '164k' surface only exists for version 'v2'
    version : {'v1, 'v2'}, optional
        Which version of the CIVET surfaces to use. Default: 'v2'

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['mid', 'white'], where corresponding
        values are Surface namedtuples containing filepaths for the left (L)
        and right (R) hemisphere surface files in OBJ format. Note: for version
        'v1', the 'mid' and 'white' files are identical.

    Other Parameters
    ----------------
    force : bool, optional
        If True, will overwrite existing dataset. Default: False
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    The CIVET template surfaces are provided in OBJ format and registered to
    ICBM152 stereotaxic space.

    The returned files include:

    - **mid**: Midthickness surface (.obj), representing the surface halfway
      between white and gray matter boundaries. For version 'v1', this is
      identical to the white surface.
    - **white**: White matter surface (.obj), representing the boundary between
      white matter and gray matter.

    The vertex density varies by option: 41k (≈41k vertices) or 164k (≈164k)
    per hemisphere. The high-resolution 164k surface is only available for
    version 'v2'.

    Example directory tree:

    ::

        ~/nnt-data/tpl-civet
        ├── v1
        │   └── civet41k
        │       ├── tpl-civet_space-ICBM152_hemi-L_den-41k_mid.obj
        │       ├── tpl-civet_space-ICBM152_hemi-L_den-41k_white.obj
        │       ├── tpl-civet_space-ICBM152_hemi-R_den-41k_mid.obj
        │       └── tpl-civet_space-ICBM152_hemi-R_den-41k_white.obj
        └── v2
            ├── civet164k
            │   ├── tpl-civet_space-ICBM152_hemi-L_den-164k_mid.obj
            │   ├── tpl-civet_space-ICBM152_hemi-L_den-164k_white.obj
            │   ├── tpl-civet_space-ICBM152_hemi-R_den-164k_mid.obj
            │   └── tpl-civet_space-ICBM152_hemi-R_den-164k_white.obj
            └── civet41k
                ├── tpl-civet_space-ICBM152_hemi-L_den-41k_mid.obj
                ├── tpl-civet_space-ICBM152_hemi-L_den-41k_white.obj
                ├── tpl-civet_space-ICBM152_hemi-R_den-41k_mid.obj
                └── tpl-civet_space-ICBM152_hemi-R_den-41k_white.obj

        5 directories, 12 files

    License: https://github.com/aces/CIVET_Full_Project/blob/master/LICENSE

    Examples
    --------
    Load the CIVET template surfaces:

    >>> surfaces = fetch_civet(density='41k', version='v2')  # doctest: +SKIP
    >>> surfaces.keys()  # doctest: +SKIP
    dict_keys(['mid', 'white'])

    Access the midthickness surface paths:

    >>> surfaces.mid  # doctest: +SKIP
    Surface(L=PosixPath('~/nnt-data/tpl-civet/v2/civet41k/tpl-civet_space-ICBM152_hemi-L_den-41k_mid.obj'),
            R=PosixPath('~/nnt-data/tpl-civet/v2/civet41k/tpl-civet_space-ICBM152_hemi-R_den-41k_mid.obj'))

    Load the left midthickness surface with nibabel:

    >>> import nibabel as nib  # doctest: +SKIP
    >>> vertices, faces = nib.freesurfer.read_geometry(surfaces.mid.L)  # doctest: +SKIP
    >>> print(f"Vertices: {vertices.shape}, Faces: {faces.shape}")  # doctest: +SKIP
    Vertices: (40962, 3), Faces: (81920, 3)

    References
    ----------
    .. [1] Oliver Lyttelton, Maxime Boucher, Steven Robbins, and Alan Evans. An
        unbiased iterative group registration template for cortical surface
        analysis. Neuroimage, 34(4):1535\u20131544, 2007.
    .. [2] Vladimir S Fonov, Alan C Evans, Robert C McKinstry, C Robert Almli,
        and DL Collins. Unbiased nonlinear average age-appropriate brain
        templates from birth to adulthood. NeuroImage, 47:S102, 2009.
    .. [3] Y Ad-Dab'bagh, O Lyttelton, J Muehlboeck, C Lepage, D Einarson, K
        Mok, O Ivanov, R Vincent, J Lerch, and E Fombonne. The civet
        image-processing environment: a fully automated comprehensive pipeline
        for anatomical neuroimaging research. proceedings of the 12th annual
        meeting of the organization for human brain mapping. Florence, Italy,
        pages 2266, 2006.
    """
    densities = ["41k", "164k"]
    if density not in densities:
        raise ValueError(
            f'The density of CIVET requested "{density}" does not exist. '
            f"Must be one of {densities}"
        )
    versions = ["v1", "v2"]
    if version not in versions:
        raise ValueError(
            f'The version of CIVET requested "{version}" does not exist. '
            f"Must be one of {versions}"
        )

    if version == "v1" and density == "164k":
        raise ValueError(
            'The "164k" density CIVET surface only exists for ' 'version "v2"'
        )

    dataset_name = "tpl-civet"
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ["mid", "white"]

    fetched = fetch_file(
        dataset_name,
        keys=[version, "civet" + density],
        force=force,
        data_dir=data_dir,
        verbose=verbose,
    )

    data = {
        k: SURFACE(
            fetched / f"tpl-civet_space-ICBM152_hemi-L_den-{density}_{k}.obj",
            fetched / f"tpl-civet_space-ICBM152_hemi-R_den-{density}_{k}.obj",
        )
        for k in keys
    }
    return Bunch(**data)


def fetch_civet_curated(version="civet41k", force=False, data_dir=None, verbose=1):
    """
    Download files for CIVET template.

    This dataset contains surface geometry files (white, midthickness, inflated,
    veryinflated, sphere), medial wall labels, and surface shape files (sulcal
    depth and vertex area) in GIFTI format for the CIVET template at multiple
    densities.

    If you used this data, please cite 1_, 2_, 3_, 4_.

    Parameters
    ----------
    version : {'civet41k', 'civet164k'}, optional
        Which density of the CIVET-space geometry files to fetch.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['white', 'midthickness', 'inflated',
        'veryinflated', 'sphere', 'medial', 'sulc', 'vaavg'], where
        corresponding values are Surface namedtuples containing filepaths for
        the left (L) and right (R) hemisphere files in GIFTI format.

    Other Parameters
    ----------------
    force : bool, optional
        If True, will overwrite existing dataset. Default: False
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    This function fetches curated CIVET surfaces from the neuromaps
    package (see `neuromaps.datasets.fetch_civet <https://netneurolab.github.io/neuromaps/generated/neuromaps.datasets.fetch_civet.html>`_).
    All files are provided in GIFTI format (.gii). The CIVET template is
    registered to ICBM152 stereotaxic space.

    The returned files include:

    - **white**: White matter surface geometry (.surf.gii), representing the
      boundary between white matter and gray matter.
    - **midthickness**: Midthickness surface geometry (.surf.gii), halfway
      between white and pial surfaces.
    - **inflated**: Inflated surface geometry (.surf.gii) for improved
      visualization of sulci and gyri.
    - **veryinflated**: Very inflated surface geometry (.surf.gii) providing
      additional smoothing for visualization.
    - **sphere**: Spherical surface geometry (.surf.gii) used for surface-based
      registration and applying parcellations.
    - **medial**: Medial wall mask (.label.gii) indicating vertices to exclude
      from analyses (vertices with no cortex).
    - **sulc**: Sulcal depth map (.shape.gii) providing sulcal/gyral patterns
      on the midthickness surface.
    - **vaavg**: Vertex area map (.shape.gii) representing the average vertex
      area on the midthickness surface.

    The vertex density varies by version: civet41k (≈41k vertices) and
    civet164k (≈164k) per hemisphere.

    Example directory tree:

    ::

        ~/nnt-data/tpl-civet_curated
        └── v2
            ├── civet164k
            │   ├── tpl-civet_den-164k_hemi-L_desc-nomedialwall_dparc.label.gii
            │   ├── tpl-civet_den-164k_hemi-L_desc-sulc_midthickness.shape.gii
            │   ├── tpl-civet_den-164k_hemi-L_desc-vaavg_midthickness.shape.gii
            │   ├── tpl-civet_den-164k_hemi-L_inflated.surf.gii
            │   ├── tpl-civet_den-164k_hemi-L_midthickness.surf.gii
            │   ├── tpl-civet_den-164k_hemi-L_sphere.surf.gii
            │   ├── tpl-civet_den-164k_hemi-L_veryinflated.surf.gii
            │   ├── tpl-civet_den-164k_hemi-L_white.surf.gii
            │   ├── tpl-civet_den-164k_hemi-R_desc-nomedialwall_dparc.label.gii
            │   ├── tpl-civet_den-164k_hemi-R_desc-sulc_midthickness.shape.gii
            │   ├── tpl-civet_den-164k_hemi-R_desc-vaavg_midthickness.shape.gii
            │   ├── tpl-civet_den-164k_hemi-R_inflated.surf.gii
            │   ├── tpl-civet_den-164k_hemi-R_midthickness.surf.gii
            │   ├── tpl-civet_den-164k_hemi-R_sphere.surf.gii
            │   ├── tpl-civet_den-164k_hemi-R_veryinflated.surf.gii
            │   ├── tpl-civet_den-164k_hemi-R_white.surf.gii
            │   ├── tpl-civet_space-fsaverage_den-164k_hemi-L_sphere.surf.gii
            │   ├── tpl-civet_space-fsaverage_den-164k_hemi-R_sphere.surf.gii
            │   ├── tpl-civet_space-fsLR_den-164k_hemi-L_sphere.surf.gii
            │   └── tpl-civet_space-fsLR_den-164k_hemi-R_sphere.surf.gii
            └── civet41k
                ├── README.md
                ├── tpl-civet_den-41k_hemi-L_desc-nomedialwall_dparc.label.gii
                ├── tpl-civet_den-41k_hemi-L_desc-sulc_midthickness.shape.gii
                ├── tpl-civet_den-41k_hemi-L_desc-vaavg_midthickness.shape.gii
                ├── tpl-civet_den-41k_hemi-L_inflated.surf.gii
                ├── tpl-civet_den-41k_hemi-L_midthickness.surf.gii
                ├── tpl-civet_den-41k_hemi-L_sphere.surf.gii
                ├── tpl-civet_den-41k_hemi-L_veryinflated.surf.gii
                ├── tpl-civet_den-41k_hemi-L_white.surf.gii
                ├── tpl-civet_den-41k_hemi-R_desc-nomedialwall_dparc.label.gii
                ├── tpl-civet_den-41k_hemi-R_desc-sulc_midthickness.shape.gii
                ├── tpl-civet_den-41k_hemi-R_desc-vaavg_midthickness.shape.gii
                ├── tpl-civet_den-41k_hemi-R_inflated.surf.gii
                ├── tpl-civet_den-41k_hemi-R_midthickness.surf.gii
                ├── tpl-civet_den-41k_hemi-R_sphere.surf.gii
                ├── tpl-civet_den-41k_hemi-R_veryinflated.surf.gii
                ├── tpl-civet_den-41k_hemi-R_white.surf.gii
                ├── tpl-civet_space-fsaverage_den-41k_hemi-L_sphere.surf.gii
                ├── tpl-civet_space-fsaverage_den-41k_hemi-R_sphere.surf.gii
                ├── tpl-civet_space-fsLR_den-41k_hemi-L_sphere.surf.gii
                └── tpl-civet_space-fsLR_den-41k_hemi-R_sphere.surf.gii

        3 directories, 41 files

    License: https://github.com/aces/CIVET_Full_Project/blob/master/LICENSE

    Examples
    --------
    Load the CIVET curated template surfaces:

    >>> surfaces = fetch_civet_curated(version='civet41k')  # doctest: +SKIP
    >>> surfaces.keys()  # doctest: +SKIP
    dict_keys([
        'white', 'midthickness', 'inflated', 'veryinflated',
        'sphere', 'medial', 'sulc', 'vaavg'
    ])

    Access the midthickness surface GIFTI files:

    >>> surfaces.midthickness  # doctest: +SKIP
    Surface(L=PosixPath('~/nnt-data/tpl-civet_curated/v2/civet41k/tpl-civet_den-41k_hemi-L_midthickness.surf.gii'),
            R=PosixPath('~/nnt-data/tpl-civet_curated/v2/civet41k/tpl-civet_den-41k_hemi-R_midthickness.surf.gii'))

    Load the left midthickness surface with nibabel:

    >>> import nibabel as nib  # doctest: +SKIP
    >>> gii = nib.load(surfaces.midthickness.L)  # doctest: +SKIP
    >>> vertices = gii.agg_data('pointset')  # doctest: +SKIP
    >>> faces = gii.agg_data('triangle')  # doctest: +SKIP
    >>> print(f"Vertices: {vertices.shape}, Faces: {faces.shape}")  # doctest: +SKIP
    Vertices: (40962, 3), Faces: (81920, 3)

    Load and examine the sulcal depth data:

    >>> sulc_left = nib.load(surfaces.sulc.L)  # doctest: +SKIP
    >>> sulc_data = sulc_left.agg_data()  # doctest: +SKIP
    >>> float(sulc_data.min()), float(sulc_data.max())  # doctest: +SKIP
    (-27.601072311401367, 20.54990005493164)

    References
    ----------
    .. [1] Oliver Lyttelton, Maxime Boucher, Steven Robbins, and Alan Evans. An
        unbiased iterative group registration template for cortical surface
        analysis. Neuroimage, 34(4):1535\u20131544, 2007.
    .. [2] Vladimir S Fonov, Alan C Evans, Robert C McKinstry, C Robert Almli,
        and DL Collins. Unbiased nonlinear average age-appropriate brain
        templates from birth to adulthood. NeuroImage, 47:S102, 2009.
    .. [3] Y Ad-Dab'bagh, O Lyttelton, J Muehlboeck, C Lepage, D Einarson, K
        Mok, O Ivanov, R Vincent, J Lerch, and E Fombonne. The civet
        image-processing environment: a fully automated comprehensive pipeline
        for anatomical neuroimaging research. proceedings of the 12th annual
        meeting of the organization for human brain mapping. Florence, Italy,
        pages 2266, 2006.
    .. [4] Ross D Markello, Justine Y Hansen, Zhen-Qi Liu, Vincent Bazinet,
        Golia Shafiei, Laura E Su\u00e1rez, Nadia Blostein, Jakob Seidlitz,
        Sylvain Baillet, Theodore D Satterthwaite, and others. Neuromaps:
        structural and functional interpretation of brain maps. Nature Methods,
        19(11):1472\u20131479, 2022.
    """
    versions = ["civet41k", "civet164k"]
    if version not in versions:
        raise ValueError(
            f"The version of fsaverage requested {version} does not "
            f"exist. Must be one of {versions}"
        )

    dataset_name = "tpl-civet_curated"
    _get_reference_info("tpl-civet_curated", verbose=verbose)

    keys = [
        "white",
        "midthickness",
        "inflated",
        "veryinflated",
        "sphere",
        "medial",
        "sulc",
        "vaavg",
    ]
    keys_suffix = {
        "white": "white.surf",
        "midthickness": "midthickness.surf",
        "inflated": "inflated.surf",
        "veryinflated": "veryinflated.surf",
        "sphere": "sphere.surf",
        "medial": "desc-nomedialwall_dparc.label",
        "sulc": "desc-sulc_midthickness.shape",
        "vaavg": "desc-vaavg_midthickness.shape",
    }
    version_density = {
        "civet41k": "41k",
        "civet164k": "164k",
    }
    density = version_density[version]

    fetched = fetch_file(
        dataset_name,
        keys=["v2", version],
        force=force,
        data_dir=data_dir,
        verbose=verbose,
    )

    # deal with default neuromaps directory structure in the archive
    if not fetched.exists():
        import shutil

        shutil.move(fetched.parent / "atlases/civet", fetched)
        shutil.rmtree(fetched.parent / "atlases")

    data = {
        k: SURFACE(
            fetched / f"tpl-civet_den-{density}_hemi-L_{keys_suffix[k]}.gii",
            fetched / f"tpl-civet_den-{density}_hemi-R_{keys_suffix[k]}.gii",
        )
        for k in keys
    }

    return Bunch(**data)


def fetch_conte69(force=False, data_dir=None, verbose=1):
    """
    Download files for Van Essen et al., 2012 Conte69 template.

    This dataset contains midthickness, inflated, and very inflated surface
    files in GIFTI format for the Conte69 atlas, a population-average surface
    template in fsLR32k space registered to MNI305 volumetric space.

    If you used this data, please cite 1_, 2_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['midthickness', 'inflated',
        'vinflated', 'info'], where 'midthickness', 'inflated', and
        'vinflated' are Surface namedtuples containing filepaths for the left
        (L) and right (R) hemisphere GIFTI files, and 'info' is a dictionary
        containing template metadata from template_description.json.

    Other Parameters
    ----------------
    force : bool, optional
        If True, will overwrite existing dataset. Default: False
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    The Conte69 template is a population-average surface atlas registered to
    MNI305 volumetric space using the fsLR32k mesh (approximately 32k vertices
    per hemisphere).

    The returned files include:

    - **midthickness**: Midthickness surface geometry (.surf.gii), halfway
      between white and pial surfaces.
    - **inflated**: Inflated surface geometry (.surf.gii) for improved
      visualization of sulci and gyri.
    - **vinflated**: Very inflated surface geometry (.surf.gii) providing
      additional smoothing for visualization.
    - **info**: Metadata dictionary containing template name, BIDS version,
      and references.

    Example directory tree:

    ::

        ~/nnt-data/tpl-conte69
        ├── CHANGES
        ├── template_description.json
        ├── tpl-conte69_space-MNI305_variant-fsLR32k_inflated.L.surf.gii
        ├── tpl-conte69_space-MNI305_variant-fsLR32k_inflated.R.surf.gii
        ├── tpl-conte69_space-MNI305_variant-fsLR32k_midthickness.L.surf.gii
        ├── tpl-conte69_space-MNI305_variant-fsLR32k_midthickness.R.surf.gii
        ├── tpl-conte69_space-MNI305_variant-fsLR32k_vinflated.L.surf.gii
        └── tpl-conte69_space-MNI305_variant-fsLR32k_vinflated.R.surf.gii

        0 directories, 8 files

    Examples
    --------
    Load the Conte69 template surfaces:

    >>> surfaces = fetch_conte69()  # doctest: +SKIP
    >>> surfaces.keys()  # doctest: +SKIP
    dict_keys(['midthickness', 'inflated', 'vinflated', 'info'])

    Access the midthickness surface GIFTI files:

    >>> surfaces.midthickness  # doctest: +SKIP
    Surface(L=PosixPath('~/nnt-data/tpl-conte69/tpl-conte69_space-MNI305_variant-fsLR32k_midthickness.L.surf.gii'),
            R=PosixPath('~/nnt-data/tpl-conte69/tpl-conte69_space-MNI305_variant-fsLR32k_midthickness.R.surf.gii'))

    Load the left midthickness surface with nibabel:

    >>> import nibabel as nib  # doctest: +SKIP
    >>> gii = nib.load(surfaces.midthickness.L)  # doctest: +SKIP
    >>> vertices = gii.agg_data('pointset')  # doctest: +SKIP
    >>> faces = gii.agg_data('triangle')  # doctest: +SKIP
    >>> print(f"Vertices: {vertices.shape}, Faces: {faces.shape}")  # doctest: +SKIP
    Vertices: (32492, 3), Faces: (64980, 3)

    Examine template metadata:

    >>> surfaces.info['Name']  # doctest: +SKIP
    "The 'Conte-69' template"

    References
    ----------
    .. [1] David C Van Essen, Kamil Ugurbil, Edward Auerbach, Deanna Barch,
        Timothy EJ Behrens, Richard Bucholz, Acer Chang, Liyong Chen, Maurizio
        Corbetta, Sandra W Curtiss, and others. The human connectome project: a
        data acquisition perspective. Neuroimage, 62(4):2222\u20132231, 2012.
    .. [2] David C Van Essen, Matthew F Glasser, Donna L Dierker, John Harwell,
        and Timothy Coalson. Parcellations and hemispheric asymmetries of human
        cerebral cortex analyzed on surface-based atlases. Cerebral cortex,
        22(10):2241\u20132262, 2012.
    .. [3] http://brainvis.wustl.edu/wiki/index.php//Caret:Atlases/Conte69_Atlas
    """
    dataset_name = "tpl-conte69"
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ["midthickness", "inflated", "vinflated"]

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    data = {
        k: SURFACE(
            fetched / f"tpl-conte69_space-MNI305_variant-fsLR32k_{k}.L.surf.gii",
            fetched / f"tpl-conte69_space-MNI305_variant-fsLR32k_{k}.R.surf.gii",
        )
        for k in keys
    }
    data["info"] = json.load(open(fetched / "template_description.json", "r"))

    return Bunch(**data)


def fetch_yerkes19(force=False, data_dir=None, verbose=1):
    """
    Download files for Donahue et al., 2016 Yerkes19 template.

    This dataset contains midthickness, inflated, and very inflated surface
    files in GIFTI format for the Yerkes19 macaque template in fsLR32k space.
    The Yerkes19 atlas is a population-average surface template for macaque
    monkeys derived from high-resolution anatomical scans.

    If you used this data, please cite 1_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['midthickness', 'inflated',
        'vinflated'], where corresponding values are Surface namedtuples
        containing filepaths for the left (L) and right (R) hemisphere GIFTI
        surface files.

    Other Parameters
    ----------------
    force : bool, optional
        If True, will overwrite existing dataset. Default: False
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Notes
    -----
    The Yerkes19 template is a macaque cortical surface atlas using the fsLR32k
    mesh (approximately 32k vertices per hemisphere). It was developed to
    facilitate comparative neuroanatomy studies between human and non-human
    primates.

    The returned files include:

    - **midthickness**: Midthickness surface geometry (.surf.gii), halfway
      between white and pial surfaces.
    - **inflated**: Inflated surface geometry (.surf.gii) for improved
      visualization of sulci and gyri.
    - **vinflated**: Very inflated surface geometry (.surf.gii) providing
      additional smoothing for visualization.

    Example directory tree:

    ::

        ~/nnt-data/tpl-yerkes19
        ├── tpl-yerkes19_space-fsLR32k_inflated.L.surf.gii
        ├── tpl-yerkes19_space-fsLR32k_inflated.R.surf.gii
        ├── tpl-yerkes19_space-fsLR32k_midthickness.L.surf.gii
        ├── tpl-yerkes19_space-fsLR32k_midthickness.R.surf.gii
        ├── tpl-yerkes19_space-fsLR32k_vinflated.L.surf.gii
        └── tpl-yerkes19_space-fsLR32k_vinflated.R.surf.gii

        0 directories, 6 files

    Examples
    --------
    Load the Yerkes19 template surfaces:

    >>> surfaces = fetch_yerkes19()  # doctest: +SKIP
    >>> surfaces.keys()  # doctest: +SKIP
    dict_keys(['midthickness', 'inflated', 'vinflated'])

    Access the midthickness surface GIFTI files:

    >>> surfaces.midthickness  # doctest: +SKIP
    Surface(L=PosixPath('~/nnt-data/tpl-yerkes19/tpl-yerkes19_space-fsLR32k_midthickness.L.surf.gii'),
            R=PosixPath('~/nnt-data/tpl-yerkes19/tpl-yerkes19_space-fsLR32k_midthickness.R.surf.gii'))

    Load the left midthickness surface with nibabel:

    >>> import nibabel as nib  # doctest: +SKIP
    >>> gii = nib.load(surfaces.midthickness.L)  # doctest: +SKIP
    >>> vertices = gii.agg_data('pointset')  # doctest: +SKIP
    >>> faces = gii.agg_data('triangle')  # doctest: +SKIP
    >>> print(f"Vertices: {vertices.shape}, Faces: {faces.shape}")  # doctest: +SKIP
    Vertices: (32492, 3), Faces: (64980, 3)

    References
    ----------
    .. [1] Chad J Donahue, Stamatios N Sotiropoulos, Saad Jbabdi, Moises
        Hernandez-Fernandez, Timothy E Behrens, Tim B Dyrby, Timothy Coalson,
        Henry Kennedy, Kenneth Knoblauch, David C Van Essen, and others. Using
        diffusion tractography to predict cortical connection strength and
        distance: a quantitative comparison with tracers in the monkey. Journal
        of Neuroscience, 36(25):6758\u20136770, 2016.
    .. [2] https://balsa.wustl.edu/reference/show/976nz
    """
    dataset_name = "tpl-yerkes19"
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ["midthickness", "inflated", "vinflated"]

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    data = {
        k: SURFACE(
            fetched / f"tpl-yerkes19_space-fsLR32k_{k}.L.surf.gii",
            fetched / f"tpl-yerkes19_space-fsLR32k_{k}.R.surf.gii",
        )
        for k in keys
    }

    return Bunch(**data)
