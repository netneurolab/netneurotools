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

    This dataset contains

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
        Dictionary-like object with keys ['surf'] where corresponding values
        are length-2 lists downloaded template files (each list composed of
        files for the left and right hemisphere).

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


def fetch_hcp_standards(force=False, data_dir=None, verbose=1):
    """
    Fetch HCP standard mesh atlases for converting between FreeSurfer and HCP.

    This dataset contains

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


def fetch_civet(density="41k", version="v1", force=False, data_dir=None, verbose=1):
    """
    Fetch CIVET surface files.

    This dataset contains

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
        Dictionary-like object with keys ['mid', 'white'] containing geometry
        files for CIVET surface. Note for version 'v1' the 'mid' and 'white'
        files are identical.

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
    License: https://github.com/aces/CIVET_Full_Project/blob/master/LICENSE

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


def fetch_conte69(force=False, data_dir=None, verbose=1):
    """
    Download files for Van Essen et al., 2012 Conte69 template.

    This dataset contains

    If you used this data, please cite 1_, 2_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['midthickness', 'inflated',
        'vinflated'], where corresponding values are lists of filepaths to
        downloaded template files.

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

    This dataset contains

    If you used this data, please cite 1_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['midthickness', 'inflated',
        'vinflated'], where corresponding values are lists of filepaths to
        downloaded template files.

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
