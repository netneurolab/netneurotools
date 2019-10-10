"""
Functions for fetching and generating datasets
"""

__all__ = [
    'fetch_cammoun2012', 'fetch_pauli2018', 'fetch_fsaverage', 'fetch_conte69',
    'fetch_connectome', 'available_connectomes', 'fetch_vazquez_rodriguez2019',
    'fetch_mirchi2018', 'make_correlated_xy'
]

from .fetchers import (fetch_cammoun2012, fetch_pauli2018, fetch_fsaverage,
                       fetch_conte69, fetch_connectome, available_connectomes,
                       fetch_vazquez_rodriguez2019)
from .generators import (make_correlated_xy)
from .mirchi import (fetch_mirchi2018)
