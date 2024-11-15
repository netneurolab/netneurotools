"""Utilites for loading / creating datasets."""

import json
import os
import shutil
from pathlib import Path
from collections import namedtuple
import importlib.resources


try:
    # nilearn 0.10.3
    from nilearn.datasets._utils import fetch_single_file as _fetch_file
except ImportError:
    from nilearn.datasets.utils import _fetch_file


SURFACE = namedtuple("Surface", ("L", "R"))

FREESURFER_IGNORE = [
    "unknown",
    "corpuscallosum",
    "Background+FreeSurfer_Defined_Medial_Wall",
]


def _get_data_dir(data_dir=None):
    """
    Get path to netneurotools data directory.

    Parameters
    ----------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None

    Returns
    -------
    data_dir : str
        Path to use as data directory
    """
    if data_dir is None:
        data_dir = os.environ.get("NNT_DATA", str(Path.home() / "nnt-data"))
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _decode_url(url_type, url):
    """
    Format `data` object with OSF API URL.

    Parameters
    ----------
    data : object
        If dict with a `url` key, will format OSF_API with relevant values

    Returns
    -------
    data : object
        Input data with all `url` dict keys formatted
    """
    OSF_API = "https://files.osf.io/v1/resources/{}/providers/osfstorage/{}"
    GITHUB_RELEASE_API = "https://github.com/{}/{}/archive/refs/tags/{}.tar.gz"
    ZENODO_FILE_API = "https://zenodo.org/record/{}/files/{}?download=1"

    if url_type == "osf":
        out_url = OSF_API.format(*url)
    elif url_type == "github-release":
        out_url = GITHUB_RELEASE_API.format(*url)
    elif url_type == "zenodo-file":
        out_url = ZENODO_FILE_API.format(*url)
    else:
        raise ValueError("URL type {} not recognized".format(url_type))

    return out_url


def fetch_file(dataset_name, keys=None, force=False, data_dir=None, verbose=1):
    """
    Fetch file(s) for dataset `dataset_name`.

    Parameters
    ----------
    dataset_name : str
        Name of dataset to fetch
    keys : str or list, optional
        If provided, will only fetch the specified key(s) from the dataset.
        Default: None
    force : bool, optional
        If True, will overwrite existing dataset. Default: False
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    verbose : int, optional
        Verbosity level. Default: 1
    """
    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)

    # deal with nested keys
    if keys is not None:
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            info = info[k]

    if "uncompress" in info and info["uncompress"]:
        targ_folder = data_dir / info["rel-path"]

        # check if folder exists and remove if force=True
        if targ_folder.exists():
            if force:
                shutil.rmtree(targ_folder)
                if verbose:
                    print(f"Dataset {dataset_name} already exists. Overwriting.")
            else:
                if verbose:
                    print(f"Dataset {dataset_name} already exists. Skipping download.")
                return targ_folder

        # download compressed file
        dl_fname = _fetch_file(
            _decode_url(info["url-type"], info["url"]),
            data_dir,
            resume=True,
            md5sum=info["md5"],
            verbose=verbose,
        )

        # extract contents and remove compressed file
        if info["url-type"] == "zenodo-file":
            archive_format = "zip"
        else:
            archive_format = "gztar"

        shutil.unpack_archive(dl_fname, targ_folder.parent, format=archive_format)
        os.remove(dl_fname)

        # rename folder if necessary
        if "rename-folder" in info:
            shutil.move(targ_folder.parent / info["rename-folder"], targ_folder)
        if verbose:
            print(f"Downloaded {dataset_name} to {data_dir}")
        return targ_folder
    else:
        targ_file = data_dir / dataset_name / info["file-name"]

        # check if file exists and remove if force=True
        if targ_file.exists():
            if force:
                os.remove(targ_file)
                if verbose:
                    print(f"Dataset {dataset_name} already exists. Overwriting.")
            else:
                if verbose:
                    print(f"Dataset {dataset_name} already exists. Skipping download.")
                return targ_file
        # download file
        dl_fname = _fetch_file(
            _decode_url(info["url-type"], info["url"]),
            data_dir / dataset_name,
            resume=True,
            md5sum=info["md5"],
            verbose=verbose,
        )
        # move/rename file
        shutil.move(dl_fname, targ_file)
        if verbose:
            print(f"Downloaded {dataset_name} to {data_dir}")
        return targ_file


def _load_resource_json(relative_path):
    """
    Load JSON file from package resources.

    Parameters
    ----------
    relative_path : str
        Path to JSON file relative to package resources

    Returns
    -------
    resource_json : dict
        JSON file loaded as a dictionary
    """
    # handling pkg_resources.resource_filename deprecation
    if getattr(importlib.resources, "files", None) is not None:
        f_resource = importlib.resources.files("netneurotools") / relative_path
    else:
        from pkg_resources import resource_filename

        f_resource = resource_filename("netneurotools", relative_path)

    with open(f_resource) as src:
        resource_json = json.load(src)

    return resource_json


NNT_DATASETS = _load_resource_json("datasets/datasets.json")
# NNT_DATASETS = _decode_urls(NNT_DATASETS)


def _get_dataset_info(name):
    """
    Return url and MD5 checksum for dataset `name`.

    Parameters
    ----------
    name : str
        Name of dataset

    Returns
    -------
    url : str
        URL from which to download dataset
    md5 : str
        MD5 checksum for file downloade from `url`
    """
    try:
        return NNT_DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Provided dataset {name} is not valid. "
            f"Must be one of: {sorted(NNT_DATASETS.keys())}"
        ) from None


NNT_REFERENCES = _load_resource_json("datasets/references.json")


def _get_reference_info(name, verbose=1, return_dict=False):
    """
    Return reference information for dataset `name`.

    Parameters
    ----------
    name : str
        Name of dataset

    Returns
    -------
    reference : str
        Reference information for dataset
    """
    try:
        curr_refs = NNT_REFERENCES[name]
        if verbose:
            print("Please cite the following papers if you are using this function:")
            for bib_category, bib_category_items in curr_refs.items():
                print(f"  [{bib_category}]:")
                for bib_item in bib_category_items:
                    print(f"    {bib_item['citation']}")

        if return_dict:
            return curr_refs
    except KeyError:
        raise KeyError(
            f"Provided dataset {name} is not valid. "
            f"Must be one of: {sorted(NNT_REFERENCES.keys())}"
        ) from None


def _fill_reference_json(bib_file, json_file, overwrite=False, use_defaults=False):
    """
    Fill in citation information for references in a JSON file.

    For internal use only.

    Parameters
    ----------
    bib_file : str
        Path to BibTeX file containing references
    json_file : str
        Path to JSON file containing references
    overwrite : bool, optional
        Whether to overwrite existing citation information. Default: False
    use_defaults : bool, optional
        Whether to use default paths for `bib_file` and `json_file`. Default: False

    Returns
    -------
    None
    """
    if use_defaults:
        bib_file = (
            importlib.resources.files("netneurotools") / "datasets/netneurotools.bib"
        )
        json_file = (
            importlib.resources.files("netneurotools") / "datasets/references.json"
        )

    from pybtex import PybtexEngine

    engine = PybtexEngine()

    def _get_citation(key):
        s = engine.format_from_file(
            filename=bib_file,
            style="unsrt",
            citations=[key],
            output_backend="plaintext",
        )
        return s.strip("\n").replace("[1] ", "")

    with open(json_file) as src:
        nnt_refs = json.load(src)

    for _, value in nnt_refs.items():
        for bib_category in value:
            for bib_item in value[bib_category]:
                if bib_item["bibkey"] not in ["", None]:
                    if bib_item["citation"] == "" or overwrite:
                        bib_item["citation"] = _get_citation(bib_item["bibkey"])

    with open(json_file, "w") as dst:
        json.dump(nnt_refs, dst, indent=4)


def _check_freesurfer_subjid(subject_id, subjects_dir=None):
    """
    Check that `subject_id` exists in provided FreeSurfer `subjects_dir`.

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
        FreeSurfer subject ID, as provided
    subjects_dir : str
        Full filepath to `subjects_dir`

    Raises
    ------
    FileNotFoundError
    """
    # check inputs for subjects_dir and subject_id
    if subjects_dir is None or not os.path.isdir(subjects_dir):
        try:
            subjects_dir = Path(os.environ["SUBJECTS_DIR"])
        except KeyError:
            subjects_dir = Path.cwd()
    else:
        subjects_dir = Path(subjects_dir).resolve()

    subjdir = subjects_dir / subject_id
    if not subjdir.is_dir():
        raise FileNotFoundError(
            f"Cannot find specified subject id {subject_id} in "
            f"provided subject directory {subjects_dir}."
        )

    return subject_id, subjects_dir


def _get_freesurfer_subjid(subject_id, subjects_dir=None):
    """
    Get fsaverage version `subject_id`, fetching if required.

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
    # check for FreeSurfer install w/fsaverage; otherwise, fetch required
    try:
        subject_id, subjects_dir = _check_freesurfer_subjid(subject_id, subjects_dir)
    except FileNotFoundError:
        if "fsaverage" not in subject_id:
            raise ValueError(
                f"Provided subject {subject_id} does not exist in provided "
                f"subjects_dir {subjects_dir}"
            ) from None
        from .fetch_template import fetch_fsaverage

        fetch_fsaverage(subject_id)
        subjects_dir = os.path.join(_get_data_dir(), "tpl-fsaverage")
        subject_id, subjects_dir = _check_freesurfer_subjid(subject_id, subjects_dir)

    return subject_id, subjects_dir
