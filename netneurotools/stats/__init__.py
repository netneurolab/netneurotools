"""Functions for performing statistical operations."""


from .correlation import (
    efficient_pearsonr,
    weighted_pearsonr,
    make_correlated_xy
)


from .permutation_test import (
    permtest_1samp,
    permtest_rel,
    permtest_pearsonr
)


from .regression import (
    _add_constant,
    residualize,
    get_dominance_stats
)


# from .stats_utils import ()


__all__ = [
    # correlation
    'efficient_pearsonr', 'weighted_pearsonr', 'make_correlated_xy',
    # permutation_test
    'permtest_1samp', 'permtest_rel', 'permtest_pearsonr',
    # regression
    '_add_constant', 'residualize', 'get_dominance_stats',
    # stats_utils
]
