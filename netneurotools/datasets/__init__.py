"""
Functions for fetching and generating datasets
"""

__all__ = [
    'fetch_cammoun2012', 'fetch_conte69', 'fetch_fsaverage', 'fetch_pauli2018',
    'make_correlated_xy', 'fetch_mirchi2018'
]

from .datasets import (fetch_cammoun2012, fetch_conte69, fetch_fsaverage,
                       fetch_pauli2018, make_correlated_xy)
from .mirchi import (fetch_mirchi2018)
