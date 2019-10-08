"""
Functions for fetching and generating datasets
"""

__all__ = [
    'fetch_cammoun2012', 'fetch_conte69', 'fetch_fsaverage', 'fetch_pauli2018',
    'make_correlated_xy', 'fetch_mirchi2018', 'fetch_connectome',
    'available_connectomes'
]

from .fetchers import (fetch_cammoun2012, fetch_conte69, fetch_fsaverage,
                       fetch_pauli2018, fetch_connectome,
                       available_connectomes)
from .generators import (make_correlated_xy)
from .mirchi import (fetch_mirchi2018)
