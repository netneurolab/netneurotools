"""For testing netneurotools.datasets.datasets_utils functionality."""
import os

import pytest

from netneurotools.datasets import datasets_utils as utils


@pytest.mark.parametrize('dset, expected', [
    ('atl-cammoun2012', ['fsaverage', 'fsaverage5', 'fsaverage6', 'fslr32k',
                         'MNI152NLin2009aSym', 'gcs']),
    ('tpl-conte69', ['url', 'md5']),
    ('atl-pauli2018', ['probabilistic', 'deterministic', 'info']),
    ('tpl-fsaverage', ['fsaverage' + f for f in ['', '3', '4', '5', '6']]),
    ('atl-schaefer2018', ['fsaverage', 'fsaverage6', 'fsaverage6'])
])
def test_get_dataset_info(dset, expected):
    """Test getting dataset info."""
    info = utils._get_dataset_info(dset)
    if isinstance(info, dict):
        assert all(k in info.keys() for k in expected)
    elif isinstance(info, list):
        for f in info:
            assert all(k in f.keys() for k in expected)
    else:
        assert False

    with pytest.raises(KeyError):
        utils._get_dataset_info('notvalid')


def test_get_data_dir(tmpdir):
    """Test getting data directory."""
    data_dir = utils._get_data_dir(tmpdir)
    assert os.path.isdir(data_dir)
