__all__ = [
    'load_mirchi2018', 'load_cammoun2012', 'make_correlated_xy'
]

from .cammoun import load_cammoun2012
from .datasets import make_correlated_xy
from .mirchi import load_mirchi2018
