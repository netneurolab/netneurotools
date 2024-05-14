"""For testing netneurotools.plotting.pysurfer_plotters functionality."""

import pytest
import numpy as np
from netneurotools import datasets, plotting


@pytest.mark.filterwarnings('ignore')
def test_plot_fsvertex():
    """Test plotting on a freesurfer vertex."""
    surfer = pytest.importorskip('surfer')

    data = np.random.rand(20484)
    brain = plotting.plot_fsvertex(data, subject_id='fsaverage5',
                                   offscreen=True)
    assert isinstance(brain, surfer.Brain)


@pytest.mark.filterwarnings('ignore')
def test_plot_fsaverage():
    """Test plotting on a freesurfer average brain."""
    surfer = pytest.importorskip('surfer')

    data = np.random.rand(68)
    lhannot, rhannot = datasets.fetch_cammoun2012('fsaverage5')['scale033']
    brain = plotting.plot_fsaverage(data, lhannot=lhannot, rhannot=rhannot,
                                    subject_id='fsaverage5', offscreen=True)
    assert isinstance(brain, surfer.Brain)
