__all__ = [
    'load_cammoun2012', 'fetch_cammoun2012', 'combine_cammoun_500',
    'fetch_mirchi2018', 'make_correlated_xy'
]

from .cammoun import fetch_cammoun2012, load_cammoun2012, combine_cammoun_500
from .datasets import make_correlated_xy
from .mirchi import fetch_mirchi2018
