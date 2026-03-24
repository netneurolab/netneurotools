"""Functions for working with GIFTI files."""

import os
import nibabel as nib

from netneurotools.interface.interface_utils import PARCIGNORE


def extract_gifti_labels(gifti_file, parc_ignore=PARCIGNORE):
    """
    Extract vertices and labels from GIFTI file.

    Parameters
    ----------
    gifti_file : str or os.PathLike or nib.GiftiImage
        Path to a GIFTI file or a pre-loaded GIFTI image
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

    # Load gifti image
    if isinstance(gifti_file, (str, os.PathLike)):
        image = nib.load(str(gifti_file))
    elif not isinstance(gifti_file, nib.gifti.GiftiImage):
        raise TypeError('`gifti_file` must be either a path to a GIFTI file'
                        'or a pre-loaded GIFTI image (`nib.gifti.GiftiImage`)')
    else:
        image = gifti_file

    surf_data = image.agg_data()
    label_table = image.labeltable
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
