"""For testing netneurotools.stats.correlation functionality."""

import pytest
import numpy as np
from netneurotools import stats


@pytest.mark.parametrize('x, y, expected', [
    # basic one-dimensional input
    (range(5), range(5), (1.0, 0.0)),
    # broadcasting occurs regardless of input order
    (np.stack([range(5), range(5, 0, -1)], 1), range(5),
     ([1.0, -1.0], [0.0, 0.0])),
    (range(5), np.stack([range(5), range(5, 0, -1)], 1),
     ([1.0, -1.0], [0.0, 0.0])),
    # correlation between matching columns
    (np.stack([range(5), range(5, 0, -1)], 1),
     np.stack([range(5), range(5, 0, -1)], 1),
     ([1.0, 1.0], [0.0, 0.0]))
])
def test_efficient_pearsonr(x, y, expected):
    """Test efficient_pearsonr function."""
    assert np.allclose(stats.efficient_pearsonr(x, y), expected)


def test_efficient_pearsonr_errors():
    """Test efficient_pearsonr function errors."""
    with pytest.raises(ValueError):
        stats.efficient_pearsonr(range(4), range(5))

    assert all(np.isnan(a) for a in stats.efficient_pearsonr([], []))


@pytest.mark.parametrize(
    "x, y, w, expected",
    [
        (
            np.array([3, 5, 6, 8, 3, 2, 6]),
            np.array([3, 5, 2, 8, 3, 3, 6]),
            np.array([7, 3, 3, 2, 4, 5, 7]),
            0.7356763090950997,
        )
    ],
)
def test_weighted_pearsonr(x, y, w, expected):
    """Test weighted_pearsonr function."""
    assert np.allclose(
        stats.weighted_pearsonr(x, y, w, use_numba=True),
        expected,
    )
    assert np.allclose(
        stats.weighted_pearsonr(x, y, w, use_numba=False),
        expected,
    )


@pytest.mark.parametrize('corr, size, tol, seed', [
    (0.85, (1000,), 0.05, 1234),
    (0.85, (1000, 1000), 0.05, 1234),
    ([[1, 0.5, 0.3], [0.5, 1, 0], [0.3, 0, 1]], (1000,), 0.05, 1234)
])
def test_make_correlated_xy(corr, size, tol, seed):
    """Test make_correlated_xy function."""
    out = stats.make_correlated_xy(corr=corr, size=size,
                                      tol=tol, seed=seed)
    # ensure output is expected shape
    assert out.shape[1:] == size
    assert len(out) == len(corr) if hasattr(corr, '__len__') else 2

    # check outputs are correlated within specified tolerance
    realcorr = np.corrcoef(out.reshape(len(out), -1))
    if len(realcorr) == 2 and not hasattr(corr, '__len__'):
        realcorr = realcorr[0, 1]
    assert np.all(np.abs(realcorr - corr) < tol)

    # check that seed generates reproducible values
    duplicate = stats.make_correlated_xy(corr=corr, size=size,
                                            tol=tol, seed=seed)
    assert np.allclose(out, duplicate)


@pytest.mark.parametrize('corr', [
    (1.5), (-1.5),                                   # outside range of [-1, 1]
    ([0.85]), ([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),  # not 2D / square array
    ([[0.85]]), ([[1, 0.5], [0.5, 0.5]])             # diagonal not equal to 1
])
def test_make_correlated_xy_errors(corr):
    """Test make_correlated_xy function errors."""
    with pytest.raises(ValueError):
        stats.make_correlated_xy(corr)
