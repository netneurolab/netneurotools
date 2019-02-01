__all__ = [
    'load_mirchi2018', 'load_cammoun2012', 'make_correlated_xy'
]

from .datasets import load_cammoun2012, make_correlated_xy
from .mirchi import load_mirchi2018
