"""Functions for working with CIFTI files."""

from pathlib import Path
import numpy as np
import nibabel as nib

from netneurotools.interface.interface_utils import PARCIGNORE


def describe_cifti(filename):
    """
    Print information about CIFTI file.

    Parameters
    ----------
    filename : str or Path
        Path to CIFTI file.

    Returns
    -------
    None
    """
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


def extract_cifti_volume(data, axis):
    """
    Extract volume data from CIFTI file.

    Adapted from [1]_.

    Parameters
    ----------
    data : np.ndarray
        Data from CIFTI file.
    axis : nibabel.cifti2.BrainModelAxis
        BrainModelAxis object from CIFTI file.

    Returns
    -------
    vol_data : np.ndarray
        Volume data.
    affine : np.ndarray
        Affine matrix.

    References
    ----------
    .. [1] https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
    """
    assert isinstance(axis, nib.cifti2.BrainModelAxis)

    vol_mask = axis.volume_mask
    vol_shape = axis.volume_shape
    vox_indices = tuple(axis.voxel[vol_mask].T)
    data = data.T[vol_mask]
    vol_data = np.zeros(vol_shape + data.shape[1:], dtype=data.dtype)
    vol_data[vox_indices] = data
    return vol_data, axis.affine


def extract_cifti_surface(data, axis, surf_name):
    """
    Extract surface data from CIFTI file.

    Adapted from [1]_.

    Parameters
    ----------
    data : np.ndarray
        Data from CIFTI file.
    axis : nibabel.cifti2.BrainModelAxis
        BrainModelAxis object from CIFTI file.
    surf_name : str
        Name of surface.

    Returns
    -------
    surf_data : np.ndarray
        Surface data.

    References
    ----------
    .. [1] https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
    """
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


def extract_cifti_labels(axis, parc_ignore=PARCIGNORE):
    """
    Extract label data from CIFTI file.

    Parameters
    ----------
    axis : nibabel.cifti2.LabelAxis
        LabelAxis object from CIFTI file.
    parc_ignore : list, optional
        List of labels to ignore.

    Returns
    -------
    keys : tuple
        Keys.
    labels : tuple
        Labels.
    """
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


def extract_cifti_surface_labels(
        data, label_axis, brainmodel_axis, surf_name, parc_ignore=PARCIGNORE):
    """
    Extract surface data and related labels from CIFTI file.

    Parameters
    ----------
    data : np.ndarray
        Data from CIFTI file.
    label_axis : nibabel.cifti2.LabelAxis
        LabelAxis object from CIFTI file.
    brainmodel_axis : nibabel.cifti2.BrainModelAxis
        BrainModelAxis object from CIFTI file.
    surf_name : str
        Name of surface.
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
    keys, labels = extract_cifti_labels(label_axis, parc_ignore=parc_ignore)

    surf_data = extract_cifti_surface(data, brainmodel_axis, surf_name)

    keys, labels = zip(
        *[
            (key, label)
            for key, label in zip(keys, labels)
            if key in np.unique(surf_data).astype(int)
        ]
    )

    surf_data = surf_data.squeeze()
    keys = tuple(map(int, keys))
    return surf_data, keys, labels


def deconstruct_cifti(filename, brain_model_axis_index=1):
    """
    Extract volume and surface data from CIFTI file.

    Adapted from [1]_.

    Parameters
    ----------
    filename : str or Path
        Path to CIFTI file.
    brain_model_axis_index : int, optional
        Index of BrainModelAxis in CIFTI file.

    Returns
    -------
    vol_data : np.ndarray
        Volume data.
    surf_left : np.ndarray
        Left hemisphere surface (``CIFTI_STRUCTURE_CORTEX_LEFT``) data.
    surf_right : np.ndarray
        Right hemisphere surface (``CIFTI_STRUCTURE_CORTEX_RIGHT``) data.


    References
    ----------
    .. [1] https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
    """
    cifti = nib.load(filename)
    data = cifti.get_fdata(dtype=np.float32)
    brain_model_axis = cifti.header.get_axis(brain_model_axis_index)
    if not isinstance(brain_model_axis, nib.cifti2.BrainModelAxis):
        raise ValueError(f"Axis {brain_model_axis_index} is not a BrainModelAxis axis.")
    return (
        extract_cifti_volume(data, brain_model_axis),
        extract_cifti_surface(data, brain_model_axis, "CIFTI_STRUCTURE_CORTEX_LEFT"),
        extract_cifti_surface(data, brain_model_axis, "CIFTI_STRUCTURE_CORTEX_RIGHT"),
    )
