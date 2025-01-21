"""For testing netneurotools.stats.permutation_test functionality."""

import pytest
import numpy as np
from netneurotools import stats

from netneurotools.stats import sw_nest_perm_ols, sw_nest


@pytest.mark.xfail
def test_permtest_1samp():
    """Test permutation test for one-sample t-test."""
    assert False
    # n1, n2, n3 = 10, 15, 20
    # rs = np.random.RandomState(1234)
    # rvn1 = rs.normal(loc=8, scale=10, size=(n1, n2, n3))

    # t1, p1 = stats.permtest_1samp(rvn1, 1, axis=0)


def test_permtest_rel():
    """Test permutation test for paired samples."""
    dr, pr = -0.0005, 0.4175824175824176
    dpr = ([dr, -dr], [pr, pr])

    rvs1 = np.linspace(1, 100, 100)
    rvs2 = np.linspace(1.01, 99.989, 100)
    rvs1_2D = np.array([rvs1, rvs2])
    rvs2_2D = np.array([rvs2, rvs1])

    # the p-values in these two cases should be consistent
    d, p = stats.permtest_rel(rvs1, rvs2, axis=0, seed=1234)
    assert np.allclose([d, p], (dr, pr))
    d, p = stats.permtest_rel(rvs1_2D.T, rvs2_2D.T, axis=0, seed=1234)
    assert np.allclose([d, p], dpr)

    # but the p-value will differ here because of _how_ we're drawing the
    # random permutations... it would be nice if this was consistent, but as
    # yet i don't have a great idea on how to make that happen without assuming
    # a whole lot about the data
    pr = 0.51248751
    tpr = ([dr, -dr], [pr, pr])
    d, p = stats.permtest_rel(rvs1_2D, rvs2_2D, axis=1, seed=1234)
    assert np.allclose([d, p], tpr)


def test_permtest_pearsonr():
    """Test permutation test for Pearson correlation."""
    np.random.seed(12345678)
    x, y = stats.make_correlated_xy(corr=0.1, size=100)
    r, p = stats.permtest_pearsonr(x, y)
    assert np.allclose([r, p], [0.10032564626876286, 0.3046953046953047])

    x, y = stats.make_correlated_xy(corr=0.5, size=100)
    r, p = stats.permtest_pearsonr(x, y)
    assert np.allclose([r, p], [0.500040365781984, 0.000999000999000999])

    z = x + np.random.normal(loc=1, size=100)
    r, p = stats.permtest_pearsonr(x, np.column_stack([y, z]))
    assert np.allclose(r, np.array([0.50004037, 0.25843187]))
    assert np.allclose(p, np.array([0.000999, 0.01098901]))

    a, b = stats.make_correlated_xy(corr=0.9, size=100)
    r, p = stats.permtest_pearsonr(np.column_stack([x, a]),
                                   np.column_stack([y, b]))
    assert np.allclose(r, np.array([0.50004037, 0.89927523]))
    assert np.allclose(p, np.array([0.000999, 0.000999]))


def test_sw_nest():
    """Test the Network Enrichment Significance Testing."""
    rng_data = np.random.default_rng(1234)

    n_subj = 100
    n_vertices = 50
    n_covariates = 2
    n_perm = 10

    observed_vars = rng_data.random((n_subj, n_vertices))
    predictor_vars = rng_data.random(n_subj)
    covariate_vars = rng_data.random((n_subj, n_covariates))
    network_ind = rng_data.choice([0, 1], size=(n_vertices,))

    # freedman_lane=False
    empirical_nofl, permuted_nofl = sw_nest_perm_ols(
        observed_vars=observed_vars,
        predictor_vars=predictor_vars,
        covariate_vars=covariate_vars,
        freedman_lane=False,
        n_perm=n_perm,
        rng=np.random.default_rng(1234),
    )
    p_nofl = sw_nest(
        empirical_nofl, permuted_nofl, network_ind
    )

    assert empirical_nofl.shape == (n_vertices,)
    assert permuted_nofl.shape == (n_perm, n_vertices)

    assert np.allclose(np.mean(empirical_nofl), -0.0043906149291564785)
    assert np.allclose(np.std(empirical_nofl), 0.11346599196523623)

    assert np.allclose(np.mean(permuted_nofl), 0.0006606492690880903)
    assert np.allclose(np.std(permuted_nofl), 0.10123368863376724)

    assert np.allclose(p_nofl, 0.6363636363636364)

    # freedman_lane=True
    empirical_fl, permuted_fl = sw_nest_perm_ols(
        observed_vars=observed_vars,
        predictor_vars=predictor_vars,
        covariate_vars=covariate_vars,
        freedman_lane=True,
        n_perm=n_perm,
        rng=np.random.default_rng(1234),
    )
    p_fl = sw_nest(
        empirical_fl, permuted_fl, network_ind
    )

    assert empirical_fl.shape == (n_vertices,)
    assert permuted_fl.shape == (n_perm, n_vertices)

    assert np.allclose(np.mean(empirical_fl), -0.0043906149291564785)
    assert np.allclose(np.std(empirical_fl), 0.11346599196523623)

    assert np.allclose(np.mean(permuted_fl), 0.004050601542834945)
    assert np.allclose(np.std(permuted_fl), 0.10545332073703956)

    assert np.allclose(p_fl, 0.5454545454545454)
