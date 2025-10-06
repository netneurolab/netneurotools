"""Transforms for datasets."""

import numpy as np
import nibabel as nib

from netneurotools.interface.cifti import extract_cifti_surface_labels
from netneurotools.interface.freesurfer import extract_annot_labels
from netneurotools.interface.gifti import extract_gifti_labels
from netneurotools.interface.interface_utils import PARCIGNORE


def _check_vertices_to_parcels_parc_file(parc_file, hemi):
    if hemi == "both":
        if not isinstance(parc_file, (tuple, list)):
            if not str(parc_file).endswith("dlabel.nii"):
                raise ValueError(
                    "parc_file should be tuple or list unless it is a dlabel.nii file"
                )
    elif hemi in ["L", "R"]:
        pass
    else:
        raise ValueError(f"Unknown hemisphere: {hemi}")

    def _check_file_type(path):
        if not str(path).endswith(("gii", "dlabel.nii", "annot")):
            raise ValueError(
                "Unsupported parcellation file format, "
                "only gii, dlabel.nii, and annot are supported."
            )

    if isinstance(parc_file, (tuple, list)):
        _check_file_type(parc_file[0])
        _check_file_type(parc_file[1])
    else:
        _check_file_type(parc_file)


def load_surf_parc_file(parc_file, cifti_structure=None, parc_ignore=PARCIGNORE):
    """
    Load surface parcellation file with vertices and labels.

    Parameters
    ----------
    parc_file : str or Path
        Path to parcellation file.
    cifti_structure : str, optional
        CIFTI structure for dlabel.nii files.
    parc_ignore : list, optional
        List of labels to ignore.

    Returns
    -------
    surf_data : np.ndarray
        Surface data.
    keys : tuple
        Keys.
    labels : tuple
        Labels.
    """
    if str(parc_file).endswith("gii"):
        surf_data, keys, labels = extract_gifti_labels(
            parc_file, parc_ignore=parc_ignore
        )
    elif str(parc_file).endswith("dlabel.nii"):
        if cifti_structure is None:
            raise ValueError("cifti_structure must be specified for dlabel.nii files")
        else:
            cifti = nib.load(parc_file)
            surf_data, keys, labels = extract_cifti_surface_labels(
                cifti.get_fdata(),
                cifti.header.get_axis(0),
                cifti.header.get_axis(1),
                cifti_structure,
                parc_ignore=parc_ignore,
            )
    elif str(parc_file).endswith("annot"):
        surf_data, keys, labels = extract_annot_labels(
            parc_file, parc_ignore=parc_ignore
        )
    else:
        raise ValueError(
            "Unsupported parcellation file format, "
            "only gii, dlabel.nii, and annot are supported."
        )
    return surf_data, keys, labels


def _vertices_to_parcels_single_hemi(
    vert_data, parc_file, cifti_structure=None, background=None, parc_ignore=PARCIGNORE
):
    vertices, keys, labels = load_surf_parc_file(
        parc_file, cifti_structure=cifti_structure, parc_ignore=parc_ignore
    )

    if vertices.shape[0] != len(vert_data):
        raise ValueError(
            "Number of vertices in provided annotation files "
            "differs from size of vertex-level data array.\n"
            "    EXPECTED: {} vertices\n"
            "    RECEIVED: {} vertices".format(vertices.shape[0], len(vert_data))
        )

    if background is not None:
        vert_data[vert_data == background] = np.nan

    reduced = []
    for idx in keys:
        found = vertices == idx
        if not found.any():
            reduced.append(np.empty_like(vert_data[0]) * np.nan)
        else:
            reduced.append(np.nanmean(vert_data[found], axis=0))
    return np.array(reduced), keys, labels


def vertices_to_parcels(
    vert_data, parc_file, hemi="both", background=None, parc_ignore=PARCIGNORE
):
    """
    Convert vertex-level data to parcel-level data.

    Parameters
    ----------
    vert_data : np.ndarray or tuple or list
        Vertex-level data.
    parc_file : str or Path or tuple or list
        Path to parcellation file.
    hemi : str, optional
        Hemisphere to process. Can be 'both', 'L', or 'R'.
    background : int, optional
        Background value in vert_data to ignore.
    parc_ignore : list, optional
        List of labels to ignore.

    Returns
    -------
    reduced : np.ndarray or tuple
        Reduced data.
    keys : tuple
        Keys.
    labels : tuple
        Labels.

    Notes
    -----
    If hemi is 'both':

    * ``vert_data`` can be a tuple or list of two arrays. It can also be a single
      array with left then right hemisphere data.
    * ``parc_file`` should be a tuple or list of two paths, unless it is a dlabel.nii
      file, in which case it can be a single path.
    * Returns tuples of reduced data, keys, and labels.

    If hemi is 'L' or 'R':

    * ``vert_data`` should be a single array.
    * ``parc_file`` should be a single path.
    * Returns reduced data, keys, and labels.
    """
    # _check_vertices_to_parcels_parc_file(parc_file, hemi)

    # deal with parc_file
    if hemi == "both":  # both hemispheres
        if isinstance(parc_file, (tuple, list)):
            cifti_structure_lh = cifti_structure_rh = None
        else:
            cifti_structure_lh = "CIFTI_STRUCTURE_CORTEX_LEFT"
            cifti_structure_rh = "CIFTI_STRUCTURE_CORTEX_RIGHT"
            parc_file = (parc_file, parc_file)

        if len(vert_data) != 2:  # not a tuple or list of two arrays
            vert_data = (
                vert_data[: len(vert_data) // 2],
                vert_data[len(vert_data) // 2 :],
            )

        reduced_lh, keys_lh, labels_lh = _vertices_to_parcels_single_hemi(
            vert_data[0],
            parc_file[0],
            cifti_structure=cifti_structure_lh,
            background=background,
            parc_ignore=parc_ignore,
        )
        reduced_rh, keys_rh, labels_rh = _vertices_to_parcels_single_hemi(
            vert_data[1],
            parc_file[1],
            cifti_structure=cifti_structure_rh,
            background=background,
            parc_ignore=parc_ignore,
        )

        reduced = (reduced_lh, reduced_rh)
        keys = (keys_lh, keys_rh)
        labels = (labels_lh, labels_rh)
    else:
        # single hemisphere
        if hemi == "L":
            cifti_structure_lh = "CIFTI_STRUCTURE_CORTEX_LEFT"
        elif hemi == "R":
            cifti_structure_lh = "CIFTI_STRUCTURE_CORTEX_RIGHT"
        else:
            raise ValueError(f"Unknown hemisphere: {hemi}")

        reduced, keys, labels = _vertices_to_parcels_single_hemi(
            vert_data,
            parc_file,
            cifti_structure=cifti_structure_lh,
            background=background,
            parc_ignore=parc_ignore,
        )

    return reduced, keys, labels


def _parcels_to_vertices_single_hemi(
    parc_data, parc_file, cifti_structure=None, fill=np.nan, parc_ignore=PARCIGNORE
):
    vertices, keys, labels = load_surf_parc_file(
        parc_file, cifti_structure=cifti_structure, parc_ignore=parc_ignore
    )

    # check if keys is ordered
    if not np.array_equal(keys, np.sort(keys)):
        raise RuntimeWarning(f"Keys are not ordered: {keys}, {labels}")

    projected = np.full(
        (vertices.shape[0],) + parc_data.shape[1:], fill, dtype=np.float64
    )
    for i_idx, idx in enumerate(keys):
        found = vertices == idx
        if not found.any():
            continue
        projected[found] = parc_data[i_idx]
    return projected, keys, labels


def parcels_to_vertices(
    parc_data, parc_file, hemi="both", fill=np.nan, parc_ignore=PARCIGNORE
):
    """
    Convert parcel-level data to vertex-level data.

    Parameters
    ----------
    parc_data : np.ndarray or tuple or list
        Parcel-level data.
    parc_file : str or Path or tuple or list
        Path to parcellation file.
    hemi : str, optional
        Hemisphere to process. Can be 'both', 'L', or 'R'.
    fill : int or float, optional
        Fill value for vertices not in parcels. Default is np.nan.
    parc_ignore : list, optional
        List of labels to ignore.

    Returns
    -------
    projected : np.ndarray or tuple
        Projected data.
    keys : tuple
        Keys.
    labels : tuple
        Labels.

    Notes
    -----
    If hemi is 'both':

    * ``parc_data`` can be a tuple or list of two arrays. It can also be a single
      array with left then right hemisphere data.
    * ``parc_file`` should be a tuple or list of two paths, unless it is a dlabel.nii
      file, in which case it can be a single path.
    * Returns tuples of projected data, keys, and labels.

    If hemi is 'L' or 'R':

    * ``parc_data`` should be a single array.
    * ``parc_file`` should be a single path.
    * Returns projected data, keys, and labels.
    """
    if hemi == "both":
        if isinstance(parc_file, (tuple, list)):
            cifti_structure_lh = cifti_structure_rh = None
        else:
            cifti_structure_lh = "CIFTI_STRUCTURE_CORTEX_LEFT"
            cifti_structure_rh = "CIFTI_STRUCTURE_CORTEX_RIGHT"
            parc_file = (parc_file, parc_file)

        projected_lh, keys_lh, labels_lh = _parcels_to_vertices_single_hemi(
            parc_data[0],
            parc_file[0],
            cifti_structure=cifti_structure_lh,
            fill=fill,
            parc_ignore=parc_ignore,
        )
        projected_rh, keys_rh, labels_rh = _parcels_to_vertices_single_hemi(
            parc_data[1],
            parc_file[1],
            cifti_structure=cifti_structure_rh,
            fill=fill,
            parc_ignore=parc_ignore,
        )

        projected = (projected_lh, projected_rh)
        keys = (keys_lh, keys_rh)
        labels = (labels_lh, labels_rh)
    else:
        if hemi == "L":
            cifti_structure_lh = "CIFTI_STRUCTURE_CORTEX_LEFT"
        elif hemi == "R":
            cifti_structure_lh = "CIFTI_STRUCTURE_CORTEX_RIGHT"
        else:
            raise ValueError(f"Unknown hemisphere: {hemi}")

        projected, keys, labels = _parcels_to_vertices_single_hemi(
            parc_data,
            parc_file,
            cifti_structure=cifti_structure_lh,
            fill=fill,
            parc_ignore=parc_ignore,
        )

    return projected, keys, labels
