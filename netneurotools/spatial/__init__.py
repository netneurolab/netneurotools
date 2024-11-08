"""Functions for handling spatial brain data."""


from .spatial_stats import (
    morans_i, local_morans_i
)


__all__ = [
    # spatial_stats
    'morans_i', 'local_morans_i'
]
