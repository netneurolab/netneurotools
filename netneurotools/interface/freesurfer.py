"""Functions for working with FreeSurfer data and parcellations."""

import numpy as np
import nibabel as nib
from netneurotools.interface.interface_utils import PARCIGNORE


def extract_annot_labels(annot_file, parc_ignore=PARCIGNORE):
    """
    Extract vertices and labels from FreeSurfer annotation file.

    Parameters
    ----------
    annot_file : str or Path
        Path to FreeSurfer annotation file.
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
    surf_data, _, labels = nib.freesurfer.read_annot(annot_file)
    labels = [label.decode() for label in labels]
    keys = np.sort(np.unique(surf_data))
    keys, labels = zip(
        *[
            (key, label)
            for key, label in zip(keys, labels)
            if label not in parc_ignore
        ]
    )
    keys = tuple(map(int, keys))
    return surf_data, keys, labels
