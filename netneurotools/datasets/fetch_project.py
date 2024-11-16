"""Functions for fetching project data."""

import os
import numpy as np

from sklearn.utils import Bunch

from .datasets_utils import _get_data_dir, _get_reference_info, fetch_file

from ._mirchi2018 import _get_fc, _get_panas


def fetch_vazquez_rodriguez2019(force=False, data_dir=None, verbose=1):
    """
    Download files from Vazquez-Rodriguez et al., 2019, PNAS.

    This dataset contains one file: rsquared_gradient.csv, which contains
    two columns: rsquared and gradient.

    If you used this data, please cite [1]_.

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object with fetched data.

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
    .. [1] Bertha V\u00e1zquez-Rodr\u00edguez, Laura E Su\u00e1rez, Ross D
        Markello, Golia Shafiei, Casey Paquola, Patric Hagmann, Martijn P Van
        Den Heuvel, Boris C Bernhardt, R Nathan Spreng, and Bratislav Misic.
        Gradients of structure\u2013function tethering across neocortex.
        Proceedings of the National Academy of Sciences,
        116(42):21219\u201321227, 2019.
    """
    dataset_name = "ds-vazquez_rodriguez2019"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    # load data
    rsq, grad = np.loadtxt(
        fetched / "rsquared_gradient.csv", delimiter=",", skiprows=1
    ).T
    data = {"rsquared": rsq, "gradient": grad}

    return Bunch(**data)


def fetch_mirchi2018(force=False, data_dir=None, verbose=1):
    """
    Download (and creates) dataset for replicating Mirchi et al., 2018, SCAN.

    Parameters
    ----------
    data_dir : str, optional
        Directory to check for existing data files (if they exist) or to save
        generated data files. Files should be named mirchi2018_fc.npy and
        mirchi2018_panas.csv for the functional connectivity and behavioral
        data, respectively.

    Returns
    -------
    X : (73, 198135) numpy.ndarray
        Functional connections from MyConnectome rsfMRI time series data
    Y : (73, 13) numpy.ndarray
        PANAS subscales from MyConnectome behavioral data
    """
    data_dir = _get_data_dir(data_dir=data_dir) / "ds-mirchi2018"
    data_dir.mkdir(exist_ok=True, parents=True)

    X_fname = data_dir / "myconnectome_fc.npy"
    Y_fname = data_dir / "myconnectome_panas.csv"

    if not os.path.exists(X_fname):
        X = _get_fc(verbose=verbose)
        np.save(X_fname, X, allow_pickle=False)
    else:
        X = np.load(X_fname, allow_pickle=False)

    if not os.path.exists(Y_fname):
        Y = _get_panas()
        np.savetxt(
            Y_fname,
            np.column_stack(list(Y.values())),
            header=",".join(Y.keys()),
            delimiter=",",
            fmt="%i",
        )
        # convert dictionary to structured array before returning
        Y = np.array(
            [tuple(row) for row in np.column_stack(list(Y.values()))],
            dtype=dict(names=list(Y.keys()), formats=["i8"] * len(Y)),
        )
    else:
        Y = np.genfromtxt(Y_fname, delimiter=",", names=True, dtype=int)

    return X, Y


def fetch_hansen_manynetworks(force=False, data_dir=None, verbose=1):
    """
    Download files from Hansen et al., 2023, PLOS Biology.

    This dataset contains

    If you used this data, please cite [1]_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with fetched data.

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
    .. [1]
    """
    dataset_name = "ds-hansen_manynetworks"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    # _parc = {
    #     "cammoun033": "Cammoun033",
    #     "schaefer100": "Schaefer100",
    #     "schaefer400": "Schaefer400",
    # }

    # data = {
    #     parc: {
    #         "bigbrain": fetched
    #         / f"data/{_parc[parc]}/bigbrain_intensities.csv",
    #         "cognitive": fetched
    #         / f"data/{_parc[parc]}/cognitive_similarity.npy",
    #         "sc": fetched / f"data/{_parc[parc]}/consensusSC.npy",
    #         "sc_wei": fetched / f"data/{_parc[parc]}/consensusSC_wei.npy",
    #         "ephys": fetched
    #         / f"data/{_parc[parc]}/electrophysiological_connectivity.npy",
    #         "gene": fetched / f"data/{_parc[parc]}/gene_coexpression.npy",
    #         "megfc": fetched
    #         / f"data/{_parc[parc]}/groupFCmeg_aec_orth_{_parc[parc]}.npy.npz",
    #         "haemodynamic": fetched
    #         / f"data/{_parc[parc]}/haemodynamic_connectivity.npy",
    #         "laminar": fetched / f"data/{_parc[parc]}/laminar_similarity.npy",
    #         "metabolic": fetched
    #         / f"data/{_parc[parc]}/metabolic_connectivity.npy",
    #         "receptor": fetched
    #         / f"data/{_parc[parc]}/receptor_similarity.npy",
    #         "temporal": fetched
    #         / f"data/{_parc[parc]}/temporal_similarity.npy",
    #         "voneconomo": fetched
    #         / f"data/{_parc[parc]}/voneconomo_{_parc[parc]}.csv",
    #     }
    #     for parc in ["cammoun033", "schaefer100", "schaefer400"]
    # }

    return fetched


def fetch_hansen_receptors(force=False, data_dir=None, verbose=1):
    """
    Download files from Hansen et al., 2022, Nature Neuroscience.

    This dataset contains

    If you used this data, please cite [1]_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with fetched data.

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
    .. [1]
    """
    dataset_name = "ds-hansen_receptors"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    return fetched


def fetch_hansen_genescognition(force=False, data_dir=None, verbose=1):
    """Download files from Hansen et al., 2021, Nature Human Behaviour.

    This dataset contains

    If you used this data, please cite [1]_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with fetched data.

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
    .. [1]
    """
    dataset_name = "ds-hansen_genescognition"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    return fetched


def fetch_hansen_brainstemfc(force=False, data_dir=None, verbose=1):
    """Download files from Hansen et al., 2024.

    This dataset contains

    If you used this data, please cite [1]_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with fetched data.

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
    .. [1]
    """
    dataset_name = "ds-hansen_brainstemfc"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    return fetched


def fetch_shafiei_megfmrimapping(force=False, data_dir=None, verbose=1):
    """Download files from Shafiei et al., 2022.

    This dataset contains

    If you used this data, please cite [1]_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with fetched data.

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
    .. [1]
    """
    dataset_name = "ds-shafiei_megfmrimapping"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    return fetched


def fetch_shafiei_megdynamics(force=False, data_dir=None, verbose=1):
    """Download files from Shafiei et al., 2023.

    This dataset contains

    If you used this data, please cite [1]_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with fetched data.

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
    .. [1]
    """
    dataset_name = "ds-shafiei_megdynamics"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    return fetched


def fetch_suarez_mami(force=False, data_dir=None, verbose=1):
    """Download files from Suarez et al., 2022, eLife.

    This dataset contains

    If you used this data, please cite [1]_.

    Returns
    -------
    filenames : :class:`sklearn.utils.Bunch`
        Dictionary-like object with fetched data.

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
    .. [1]
    """
    dataset_name = "ds-suarez_mami"
    _get_reference_info(dataset_name, verbose=verbose)

    fetched = fetch_file(dataset_name, force=force, data_dir=data_dir, verbose=verbose)

    return fetched


def fetch_famous_gmat(dataset, force=False, data_dir=None, verbose=1):
    """
    Download files from multi-species connectomes.

    This dataset contains

    If you used this data, please cite celegans [1]_, drosophila [2]_, human
    [3]_, macaque_markov [4]_, macaque_modha [5]_, mouse [6]_, rat [7]_.

    Parameters
    ----------
    dataset : str
        Specifies which dataset to download.

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        Dictionary-like object with, at a minimum, keys ['conn', 'labels',
        'ref'] providing connectivity / correlation matrix, region labels, and
        relevant reference. Other possible keys include 'dist' (an array of
        Euclidean distances between regions of 'conn'), 'coords' (an array of
        xyz coordinates for regions of 'conn'), 'acronyms' (an array of
        acronyms for regions of 'conn'), and 'networks' (an array of network
        affiliations for regions of 'conn').

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
    .. [1] Lav R Varshney, Beth L Chen, Eric Paniagua, David H Hall, and Dmitri
        B Chklovskii. Structural properties of the caenorhabditis elegans
        neuronal network. PLoS computational biology, 7(2):e1001066, 2011.
    .. [2] Ann-Shyn Chiang, Chih-Yung Lin, Chao-Chun Chuang, Hsiu-Ming Chang,
        Chang-Huain Hsieh, Chang-Wei Yeh, Chi-Tin Shih, Jian-Jheng Wu, Guo-Tzau
        Wang, Yung-Chang Chen, and others. Three-dimensional reconstruction of
        brain-wide wiring networks in drosophila at single-cell resolution.
        Current biology, 21(1):1\u201311, 2011.
    .. [3] Alessandra Griffa, Yasser Alem\u00e1n-G\u00f3mez, and Patric Hagmann.
        Structural and functional connectome from 70 young healthy adults [data
        set]. Zenodo, 2019.
    .. [4] Nikola T Markov, Maria Ercsey-Ravasz, Camille Lamy, Ana Rita Ribeiro
        Gomes, Lo\u00efc Magrou, Pierre Misery, Pascale Giroud, Pascal Barone,
        Colette Dehay, Zolt\u00e1n Toroczkai, and others. The role of long-range
        connections on the specificity of the macaque interareal cortical
        network. Proceedings of the National Academy of Sciences,
        110(13):5187\u20135192, 2013.
    .. [5] Dharmendra S Modha and Raghavendra Singh. Network architecture of the
        long-distance pathways in the macaque brain. Proceedings of the National
        Academy of Sciences, 107(30):13485\u201313490, 2010.
    .. [6] Mikail Rubinov, Rolf JF Ypma, Charles Watson, and Edward T Bullmore.
        Wiring cost and topological participation of the mouse brain connectome.
        Proceedings of the National Academy of Sciences,
        112(32):10032\u201310037, 2015.
    .. [7] Mihail Bota, Olaf Sporns, and Larry W Swanson. Architecture of the
        cerebral cortical association connectome underlying cognition.
        Proceedings of the National Academy of Sciences,
        112(16):E2093\u2013E2101, 2015.
    """
    available_connectomes = [
        "celegans",
        "drosophila",
        "human_func_scale033",
        "human_func_scale060",
        "human_func_scale125",
        "human_func_scale250",
        "human_func_scale500",
        "human_struct_scale033",
        "human_struct_scale060",
        "human_struct_scale125",
        "human_struct_scale250",
        "human_struct_scale500",
        "macaque_markov",
        "macaque_modha",
        "mouse",
        "rat",
    ]

    if dataset not in available_connectomes:
        raise ValueError(
            f"Provided dataset {dataset} not available; "
            f"must be one of {available_connectomes}"
        )

    base_dataset_name = "ds-famous_gmat"
    _get_reference_info(base_dataset_name, verbose=verbose)

    fetched = fetch_file(
        base_dataset_name, force=force, data_dir=data_dir, verbose=verbose
    )

    data = {}
    for f in (fetched / dataset).glob("*.csv"):
        try:
            data[f.stem] = np.loadtxt(f, delimiter=",")
        except ValueError:
            data[f.stem] = np.loadtxt(f, delimiter=",", dtype=str)

    return Bunch(**data)


def fetch_neurosynth():
    """Download Neurosynth data."""
    pass
