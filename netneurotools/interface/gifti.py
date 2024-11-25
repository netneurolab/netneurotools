"""Functions for working with GIFTI files."""

import nibabel as nib

from netneurotools.interface.interface_utils import PARCIGNORE


def extract_gifti_labels(gifti_file, parc_ignore=PARCIGNORE):
    """
    Extract vertices and labels from GIFTI file.

    Parameters
    ----------
    gifti_file : str or Path
        Path to GIFTI file.
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
    surf_data = nib.load(gifti_file).agg_data()
    label_table = nib.load(gifti_file).labeltable
    keys, labels = list(
        zip(
            *[
                label_tuple
                for label_tuple in label_table.get_labels_as_dict().items()
                if label_tuple[1] not in parc_ignore
            ]
        )
    )
    keys = tuple(map(int, keys))
    return surf_data, keys, labels
