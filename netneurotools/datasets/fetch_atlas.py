"""Functions for fetching atlas data."""

import itertools


from sklearn.utils import Bunch

from .datasets_utils import SURFACE, _get_reference_info, fetch_file


def fetch_cammoun2012(
    version="MNI152NLin2009aSym", force=False, data_dir=None, verbose=1
):
    """
    Download files for Cammoun et al., 2012 multiscale parcellation.

    This dataset contains

    If you used this data, please cite 1_.

    Parameters
    ----------
    version : str, optional
        Specifies which version of the dataset to download, where
        'MNI152NLin2009aSym' will return .nii.gz atlas files defined in MNI152
        space, 'fsaverageX' will return .annot files defined in fsaverageX
        space (FreeSurfer 6.0.1), 'fslr32k' will return .label.gii files in
        fs_LR_32k HCP space, and 'gcs' will return FreeSurfer-style .gcs
        probabilistic atlas files for generating new, subject-specific
        parcellations. Default: 'MNI152NLin2009aSym'

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['scale033', 'scale060', 'scale125',
        'scale250', 'scale500'], where corresponding values are lists of
        filepaths to downloaded parcellation files.

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
    License: https://raw.githubusercontent.com/LTS5/cmp/master/COPYRIGHT

    References
    ----------
    .. [1] Leila Cammoun, Xavier Gigandet, Djalel Meskaldji, Jean Philippe
        Thiran, Olaf Sporns, Kim Q Do, Philippe Maeder, Reto Meuli, and Patric
        Hagmann. Mapping the human connectome at multiple scales with diffusion
        spectrum mri. Journal of neuroscience methods, 203(2):386\u2013397,
        2012.
    """
    versions = [
        "gcs",
        "fsaverage",
        "fsaverage5",
        "fsaverage6",
        "fslr32k",
        "MNI152NLin2009aSym",
    ]
    if version not in versions:
        raise ValueError(
            f"The version of Cammoun et al., 2012 parcellation "
            f"requested {version} does not exist. Must be one of {versions}"
        )

    dataset_name = "atl-cammoun2012"
    _get_reference_info(dataset_name, verbose=verbose)

    keys = ["scale033", "scale060", "scale125", "scale250", "scale500"]

    fetched = fetch_file(
        dataset_name, keys=version, force=force, data_dir=data_dir, verbose=verbose
    )

    if version == "MNI152NLin2009aSym":
        _fname = "atl-Cammoun2012_space-MNI152NLin2009aSym_res-{}_deterministic.nii.gz"
        data = {
            k: fetched
            / _fname.format(k[-3:])
            for k in keys
        }
        data["info"] = fetched / "atl-Cammoun2012_space-MNI152NLin2009aSym_info.csv"
    elif version == "fslr32k":
        _fname = "atl-Cammoun2012_space-fslr32k_res-{}_hemi-{}_deterministic.label.gii"
        data = {
            k: SURFACE(
                fetched / _fname.format(k[-3:], "L"),
                fetched / _fname.format(k[-3:], "R")
            )
            for k in keys
        }
    elif version in ("fsaverage", "fsaverage5", "fsaverage6"):
        _fname = "atl-Cammoun2012_space-{}_res-{}_hemi-{}_deterministic.annot"
        data = {
            k: SURFACE(
                fetched / _fname.format(version, k[-3:], "L"),
                fetched / _fname.format(version, k[-3:], "R")
            )
            for k in keys
        }
    else:
        data = {
            k: [
                fetched / f"atl-Cammoun2012_res-{k[5:]}_hemi-L_probabilistic.gcs",
                fetched / f"atl-Cammoun2012_res-{k[5:]}_hemi-R_probabilistic.gcs",
            ]
            for k in keys[:-1]
        }
        data[keys[-1]] = list(
            itertools.chain.from_iterable(
                [
                    [
                        fetched
                        / f"atl-Cammoun2012_res-{k[5:]}_hemi-L_probabilistic.gcs",
                        fetched
                        / f"atl-Cammoun2012_res-{k[5:]}_hemi-R_probabilistic.gcs",
                    ]
                    for k in ["scale500v1", "scale500v2", "scale500v3"]
                ]
            )
        )

    return Bunch(**data)


def fetch_schaefer2018(version="fsaverage", force=False, data_dir=None, verbose=1):
    """
    Download FreeSurfer .annot files for Schaefer et al., 2018 parcellation.

    This dataset contains

    If you used this data, please cite 1_.

    Parameters
    ----------
    version : {'fsaverage', 'fsaverage5', 'fsaverage6', 'fslr32k'}
        Specifies which surface annotation files should be matched to. Default:
        'fsaverage'

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys of format '{}Parcels{}Networks' where
        corresponding values are the left/right hemisphere annotation files

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
    License: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

    References
    ----------
    .. [1] Alexander Schaefer, Ru Kong, Evan M Gordon, Timothy O Laumann,
        Xi-Nian Zuo, Avram J Holmes, Simon B Eickhoff, and BT Thomas Yeo.
        Local-global parcellation of the human cerebral cortex from intrinsic
        functional connectivity mri. Cerebral cortex, 28(9):3095\u20133114,
        2018.
    """
    versions = ["fsaverage", "fsaverage5", "fsaverage6", "fslr32k"]
    if version not in versions:
        raise ValueError(
            f"The version of Schaefer et al., 2018 parcellation "
            f'requested "{version}" does not exist. Must be one of {versions}'
        )

    dataset_name = "atl-schaefer2018"
    _get_reference_info(dataset_name, verbose=verbose)

    keys = [f"{p}Parcels{n}Networks" for p in range(100, 1001, 100) for n in [7, 17]]

    fetched = fetch_file(
        dataset_name, keys=version, force=force, data_dir=data_dir, verbose=verbose
    )

    if version == "fslr32k":
        _fname = "atl-Schaefer2018_space-{}_hemi-{}_desc-{}_deterministic.dlabel.nii"
        data = {
            k: fetched / _fname.format(version, "LR", k)
            for k in keys
        }
    else:
        _fname = "atl-Schaefer2018_space-{}_hemi-{}_desc-{}_deterministic.annot"
        data = {
            k: SURFACE(
                fetched / _fname.format(version, "L", k),
                fetched / _fname.format(version, "R", k)
            )
            for k in keys
        }

    return Bunch(**data)


def fetch_mmpall(version="fslr32k", force=False, data_dir=None, verbose=1):
    """
    Download .label.gii files for Glasser et al., 2016 MMPAll atlas.

    This dataset contains

    If you used this data, please cite 1_.

    Parameters
    ----------
    version : {'fslr32k'}
        Specifies which surface annotation files should be matched to. Default:
        'fslr32k'

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Namedtuple with fields ('lh', 'rh') corresponding to filepaths to
        left/right hemisphere parcellation files

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
    License: https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms

    References
    ----------
    .. [1] Matthew F Glasser, Timothy S Coalson, Emma C Robinson, Carl D Hacker,
        John Harwell, Essa Yacoub, Kamil Ugurbil, Jesper Andersson, Christian F
        Beckmann, Mark Jenkinson, and others. A multi-modal parcellation of
        human cerebral cortex. Nature, 536(7615):171\u2013178, 2016.
    """
    versions = ["fslr32k"]
    if version not in versions:
        raise ValueError(
            f"The version of Glasser et al., 2016 parcellation "
            f'requested "{version}" does not exist. Must be one of {versions}'
        )

    dataset_name = "atl-mmpall"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(
        dataset_name, keys=version, force=force, data_dir=data_dir, verbose=verbose
    )

    return SURFACE(
        fetched / f"atl-MMPAll_space-{version}_hemi-L_deterministic.label.gii",
        fetched / f"atl-MMPAll_space-{version}_hemi-R_deterministic.label.gii",
    )


def fetch_pauli2018(force=False, data_dir=None, verbose=1):
    """
    Download files for Pauli et al., 2018 subcortical parcellation.

    This dataset contains

    If you used this data, please cite 1_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['probabilistic', 'deterministic'],
        where corresponding values are filepaths to downloaded atlas files.

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
    License: CC-BY Attribution 4.0 International

    References
    ----------
    .. [1] Wolfgang M Pauli, Amanda N Nili, and J Michael Tyszka. A
        high-resolution probabilistic in vivo atlas of human subcortical brain
        nuclei. Scientific data, 5(1):1\u201313, 2018.
    """
    dataset_name = "atl-pauli2018"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    data = {
        "probabilistic": fetched
        / "atl-pauli2018_space-MNI152NLin2009cAsym_hemi-both_probabilistic.nii.gz",
        "deterministic": fetched
        / "atl-pauli2018_space-MNI152NLin2009cAsym_hemi-both_deterministic.nii.gz",
        "info": fetched / "atl-pauli2018_space-MNI152NLin2009cAsym_info.csv",
    }

    return Bunch(**data)


def fetch_ye2020():
    """Fetch Ye et al., 2020 subcortical parcellation."""
    pass


def fetch_voneconomo(force=False, data_dir=None, verbose=1):
    """
    Fetch von-Economo Koskinas probabilistic FreeSurfer atlas.

    This dataset contains

    If you used this data, please cite 1_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['gcs', 'ctab', 'info']

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
    License: CC-BY-NC-SA 4.0

    References
    ----------
    .. [1] Lianne H Scholtens, Marcel A de Reus, Siemon C de Lange, Ruben
        Schmidt, and Martijn P van den Heuvel. An mri von economo\u2013koskinas
        atlas. NeuroImage, 170:249\u2013256, 2018.
    """
    dataset_name = "atl-voneconomo_koskinas"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    data = {
        "gcs": SURFACE(
            fetched / "atl-vonEconomoKoskinas_hemi-L_probabilistic.gcs",
            fetched / "atl-vonEconomoKoskinas_hemi-R_probabilistic.gcs",
        ),
        "ctab": SURFACE(
            fetched / "atl-vonEconomoKoskinas_hemi-L_probabilistic.ctab",
            fetched / "atl-vonEconomoKoskinas_hemi-R_probabilistic.ctab",
        ),
        "info": fetched / "atl-vonEconomoKoskinas_info.csv",
    }

    return Bunch(**data)
