"""Transforms for datasets."""

from pathlib import Path
import numpy as np
import nibabel as nib

PARCIGNORE = [
    "unknown",
    "corpuscallosum",
    "Background+FreeSurfer_Defined_Medial_Wall",
    "???",
    "Unknown",
    "Medial_wall",
    "Medial wall",
    "medial_wall",
]


def _describe_cifti(filename):
    if not isinstance(filename, Path):
        filename = Path(filename)

    cifti = nib.load(filename)
    print(f"Filename: {filename.name}", f"Path: {filename}", "", sep="\n")

    header = cifti.header
    n_axes = header.number_of_mapped_indices
    axes_indices = header.mapped_indices
    print(f"Number of axes: {n_axes}, axes indices: {axes_indices}")

    for i_axes in axes_indices:
        curr_axis = header.get_axis(i_axes)
        print(f"Axis {i_axes}: {curr_axis.__class__.__name__}")
        # LabelAxis
        if isinstance(curr_axis, nib.cifti2.LabelAxis):
            for k, v in curr_axis.label[0].items():
                print(f"\t{k}:\t{v[0]}\t\t{v[1]}")
        # BrainModelAxis
        if isinstance(curr_axis, nib.cifti2.BrainModelAxis):
            if curr_axis.nvertices:
                print(f"\tSURFACE: {curr_axis.nvertices}")
                for struct_name, struct_slice, _ in curr_axis.iter_structures():
                    if struct_name in [
                        "CIFTI_STRUCTURE_CORTEX_LEFT",
                        "CIFTI_STRUCTURE_CORTEX_RIGHT",
                    ]:
                        print(f"\t\t{struct_name}\t\t{struct_slice}")
            if curr_axis.volume_shape:
                print(f"\tVOLUME: {curr_axis.volume_shape}")
                for struct_name, struct_slice, _ in curr_axis.iter_structures():
                    if struct_name not in [
                        "CIFTI_STRUCTURE_CORTEX_LEFT",
                        "CIFTI_STRUCTURE_CORTEX_RIGHT",
                    ]:
                        print(f"\t\t{struct_name}\t\t{struct_slice}")


def _extract_cifti_volume(data, axis):
    """Extract volume data from CIFTI file."""
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    vol_mask = axis.volume_mask
    vol_shape = axis.volume_shape
    vox_indices = tuple(axis.voxel[vol_mask].T)
    data = data.T[vol_mask]
    vol_data = np.zeros(vol_shape + data.shape[1:], dtype=data.dtype)
    vol_data[vox_indices] = data
    return nib.Nifti1Image(vol_data, affine=axis.affine)


def _extract_cifti_surface(data, axis, surf_name):
    """Extract surface data from CIFTI file."""
    assert isinstance(axis, nib.cifti2.BrainModelAxis)

    found = [
        struct_tuple
        for struct_tuple in axis.iter_structures()
        if struct_tuple[0] == surf_name
    ]
    if len(found) == 0:
        raise ValueError(f"No structure named {surf_name}")
    elif len(found) > 1:
        raise ValueError(
            f"Multiple structures named {surf_name}", f"Structures: {found}"
        )
    else:
        name, data_indices, model = found[0]  # noqa: F841

    data = data.T[data_indices]
    vtx_indices = model.vertex
    surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
    surf_data[vtx_indices] = data

    return surf_data


def _extract_cifti_label(axis, parc_ignore=PARCIGNORE):
    assert isinstance(axis, nib.cifti2.LabelAxis)
    keys, labels = list(
        zip(
            *[
                (label_tuple[0], label_tuple[1][0])
                for label_tuple in axis.label[0].items()
                if label_tuple[1][0] not in parc_ignore
            ]
        )
    )
    return keys, labels


def _extract_cifti_surface_label(
        data, label_axis, brainmodel_axis, surf_name, parc_ignore=PARCIGNORE):
    """Extract surface data from CIFTI file."""
    keys, labels = _extract_cifti_label(label_axis, parc_ignore=parc_ignore)

    surf_data = _extract_cifti_surface(data, brainmodel_axis, surf_name)

    keys, labels = zip(
        *[
            (key, label)
            for key, label in zip(keys, labels)
            if key in np.unique(surf_data).astype(int)
        ]
    )
    return surf_data, keys, labels


def _deconstruct_cifti(filename, brain_model_axis_index=1):
    """Extract volume and surface data from CIFTI file."""
    cifti = nib.load(filename)
    data = cifti.get_fdata(dtype=np.float32)
    brain_model_axis = cifti.header.get_axis(brain_model_axis_index)
    if not isinstance(brain_model_axis, nib.cifti2.BrainModelAxis):
        raise ValueError(f"Axis {brain_model_axis_index} is not a BrainModelAxis axis.")
    return (
        _extract_cifti_volume(data, brain_model_axis),
        _extract_cifti_surface(data, brain_model_axis, "CIFTI_STRUCTURE_CORTEX_LEFT"),
        _extract_cifti_surface(data, brain_model_axis, "CIFTI_STRUCTURE_CORTEX_RIGHT"),
    )


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


def _load_parc_file(parc_file, cifti_structure=None, parc_ignore=PARCIGNORE):

    if str(parc_file).endswith("gii"):
        vertices = nib.load(parc_file).agg_data()
        keys, labels = list(
            zip(
                *[
                    label_tuple
                    for label_tuple in nib.load(parc_file)
                    .labeltable.get_labels_as_dict()
                    .items()
                    if label_tuple[1] not in parc_ignore
                ]
            )
        )
    elif str(parc_file).endswith("dlabel.nii"):
        if cifti_structure is None:
            raise ValueError("cifti_structure must be specified for dlabel.nii files")
        else:
            cifti = nib.load(parc_file)
            vertices, keys, labels = _extract_cifti_surface_label(
                cifti.get_fdata(),
                cifti.header.get_axis(0),
                cifti.header.get_axis(1),
                cifti_structure,
                parc_ignore=parc_ignore,
            )
            vertices = vertices.squeeze()
    elif str(parc_file).endswith("annot"):
        vertices, _, labels = nib.freesurfer.read_annot(parc_file)
        labels = [label.decode() for label in labels]
        keys = np.sort(np.unique(vertices))
        keys, labels = zip(
            *[
                (key, label)
                for key, label in zip(keys, labels)
                if label not in parc_ignore
            ]
        )
    else:
        raise ValueError(
            "Unsupported parcellation file format, "
            "only gii, dlabel.nii, and annot are supported."
        )
    keys = tuple(map(int, keys))
    return vertices, keys, labels


def _vertices_to_parcels_single_hemi(
    vert_data, parc_file, cifti_structure=None, background=None, parc_ignore=PARCIGNORE
):
    vertices, keys, labels = _load_parc_file(
        parc_file, cifti_structure=cifti_structure, parc_ignore=parc_ignore)

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
    """Convert vertex-level data to parcel-level data."""
    _check_vertices_to_parcels_parc_file(parc_file, hemi)

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


def _parcels_to_vertices_single_hemi(parc_data, parc_file, cifti_structure=None):
    pass


def parcels_to_vertices(parc_data, parc_file, hemi="both"):
    """Convert parcel-level data to vertex-level data."""
    pass
