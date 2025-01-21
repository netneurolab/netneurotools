"""Functions for calculating permutation test."""

import numpy as np
import scipy.stats as sstats
from sklearn.utils.validation import check_random_state

try:  # scipy >= 1.8.0
    from scipy.stats._stats_py import _chk2_asarray
except ImportError:  # scipy < 1.8.0
    from scipy.stats.stats import _chk2_asarray

try:
    import statsmodels.api as sma
except ImportError:
    _has_statsmodels = False
else:
    _has_statsmodels = True

from .correlation import efficient_pearsonr


def permtest_1samp(a, popmean, axis=0, n_perm=1000, seed=0):
    """
    Non-parametric equivalent of :py:func:`scipy.stats.ttest_1samp`.

    Generates two-tailed p-value for hypothesis of whether `a` differs from
    `popmean` using permutation tests

    Parameters
    ----------
    a : array_like
        Sample observations
    popmean : float or array_like
        Expected valued in null hypothesis. If array_like then it must have the
        same shape as `a` excluding the `axis` dimension
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole array
        of `a`. Default: 0
    n_perm : int, optional
        Number of permutations to assess. Unless `a` is very small along `axis`
        this will approximate a randomization test via Monte Carlo simulations.
        Default: 1000
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Set to None for "randomness".
        Default: 0

    Returns
    -------
    stat : float or numpy.ndarray
        Difference from `popmean`
    pvalue : float or numpy.ndarray
        Non-parametric p-value

    Notes
    -----
    Providing multiple values to `popmean` to run *independent* tests in
    parallel is not currently supported.

    The lowest p-value that can be returned by this function is equal to 1 /
    (`n_perm` + 1).

    Examples
    --------
    >>> from netneurotools import stats
    >>> np.random.seed(7654567)  # set random seed for reproducible results
    >>> rvs = np.random.normal(loc=5, scale=10, size=(50, 2))

    Test if mean of random sample is equal to true mean, and different mean. We
    reject the null hypothesis in the second case and don't reject it in the
    first case.

    >>> stats.permtest_1samp(rvs, 5.0)
    (array([-0.985602  , -0.05204969]), array([0.48551449, 0.95904096]))
    >>> stats.permtest_1samp(rvs, 0.0)
    (array([4.014398  , 4.94795031]), array([0.00699301, 0.000999  ]))

    Example using axis and non-scalar dimension for population mean

    >>> stats.permtest_1samp(rvs, [5.0, 0.0])
    (array([-0.985602  ,  4.94795031]), array([0.48551449, 0.000999  ]))
    >>> stats.permtest_1samp(rvs.T, [5.0, 0.0], axis=1)
    (array([-0.985602  ,  4.94795031]), array([0.51548452, 0.000999  ]))
    """
    a, popmean, axis = _chk2_asarray(a, popmean, axis)
    rs = check_random_state(seed)

    if a.size == 0:
        return np.nan, np.nan

    # ensure popmean will broadcast to `a` correctly
    if popmean.ndim != a.ndim:
        popmean = np.expand_dims(popmean, axis=axis)

    # center `a` around `popmean` and calculate original mean
    zeroed = a - popmean
    true_mean = zeroed.mean(axis=axis) / 1
    abs_mean = np.abs(true_mean)

    # this for loop is not _the fastest_ but is memory efficient
    # the broadcasting alt. would mean storing zeroed.size * n_perm in memory
    permutations = np.ones(true_mean.shape)
    for _ in range(n_perm):
        flipped = zeroed * rs.choice([-1, 1], size=zeroed.shape)  # sign flip
        permutations += np.abs(flipped.mean(axis=axis)) >= abs_mean

    pvals = permutations / (n_perm + 1)  # + 1 in denom accounts for true_mean

    return true_mean, pvals


def permtest_rel(a, b, axis=0, n_perm=1000, seed=0):
    """
    Non-parametric equivalent of :py:func:`scipy.stats.ttest_rel`.

    Generates two-tailed p-value for hypothesis of whether related samples `a`
    and `b` differ using permutation tests

    Parameters
    ----------
    a, b : array_like
        Sample observations. These arrays must have the same shape.
    axis : int or None, optional
        Axis along which to compute test. If None, compute over whole arrays
        of `a` and `b`. Default: 0
    n_perm : int, optional
        Number of permutations to assess. Unless `a` and `b` are very small
        along `axis` this will approximate a randomization test via Monte
        Carlo simulations. Default: 1000
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Set to None for "randomness".
        Default: 0

    Returns
    -------
    stat : float or numpy.ndarray
        Average difference between `a` and `b`
    pvalue : float or numpy.ndarray
        Non-parametric p-value

    Notes
    -----
    The lowest p-value that can be returned by this function is equal to 1 /
    (`n_perm` + 1).

    Examples
    --------
    >>> from netneurotools import stats

    >>> np.random.seed(12345678)  # set random seed for reproducible results
    >>> rvs1 = np.random.normal(loc=5, scale=10, size=500)
    >>> rvs2 = (np.random.normal(loc=5, scale=10, size=500)
    ...         + np.random.normal(scale=0.2, size=500))
    >>> stats.permtest_rel(rvs1, rvs2)  # doctest: +SKIP
    (-0.16506275161572695, 0.8021978021978022)

    >>> rvs3 = (np.random.normal(loc=8, scale=10, size=500)
    ...         + np.random.normal(scale=0.2, size=500))
    >>> stats.permtest_rel(rvs1, rvs3)  # doctest: +SKIP
    (2.40533726097883, 0.000999000999000999)
    """
    a, b, axis = _chk2_asarray(a, b, axis)
    rs = check_random_state(seed)

    if a.shape[axis] != b.shape[axis]:
        raise ValueError('Provided arrays do not have same length along axis')

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    # calculate original difference in means
    ab = np.stack([a, b], axis=0)
    if ab.ndim < 3:
        ab = np.expand_dims(ab, axis=-1)
    true_diff = np.squeeze(np.diff(ab, axis=0)).mean(axis=axis) / 1
    abs_true = np.abs(true_diff)

    # idx array
    reidx = list(np.meshgrid(*[range(f) for f in ab.shape], indexing='ij'))

    permutations = np.ones(true_diff.shape)
    for _ in range(n_perm):
        # use this to re-index (i.e., swap along) the first axis of `ab`
        swap = rs.random_sample(ab.shape[:-1]).argsort(axis=axis)
        reidx[0] = np.repeat(swap[..., np.newaxis], ab.shape[-1], axis=-1)
        # recompute difference between `a` and `b` (i.e., first axis of `ab`)
        pdiff = np.squeeze(np.diff(ab[tuple(reidx)], axis=0)).mean(axis=axis)
        permutations += np.abs(pdiff) >= abs_true

    pvals = permutations / (n_perm + 1)  # + 1 in denom accounts for true_diff

    return true_diff, pvals


def permtest_pearsonr(a, b, axis=0, n_perm=1000, resamples=None, seed=0):
    """
    Non-parametric equivalent of :py:func:`scipy.stats.pearsonr`.

    Generates two-tailed p-value for hypothesis of whether samples `a` and `b`
    are correlated using permutation tests

    Parameters
    ----------
    a,b : (N[, M]) array_like
        Sample observations. These arrays must have the same length and either
        an equivalent number of columns or be broadcastable
    axis : int or None, optional
        Axis along which to compute test. If None, compute over whole arrays
        of `a` and `b`. Default: 0
    n_perm : int, optional
        Number of permutations to assess. Unless `a` and `b` are very small
        along `axis` this will approximate a randomization test via Monte
        Carlo simulations. Default: 1000
    resamples : (N, P) array_like, optional
        Resampling array used to shuffle `a` when generating null distribution
        of correlations. This array must have the same length as `a` and `b`
        and should have at least the same number of columns as `n_perm` (if it
        has more then only `n_perm` columns will be used. When not specified a
        standard permutation is used to shuffle `a`. Default: None
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Set to None for "randomness".
        Default: 0

    Returns
    -------
    corr : float or numpyndarray
        Correlations
    pvalue : float or numpy.ndarray
        Non-parametric p-value

    Notes
    -----
    The lowest p-value that can be returned by this function is equal to 1 /
    (`n_perm` + 1).

    Examples
    --------
    >>> from netneurotools import stats

    >>> np.random.seed(12345678)  # set random seed for reproducible results
    >>> x, y = stats.make_correlated_xy(corr=0.1, size=100)
    >>> stats.permtest_pearsonr(x, y)  # doctest: +SKIP
    (0.10032564626876286, 0.3046953046953047)

    >>> x, y = stats.make_correlated_xy(corr=0.5, size=100)
    >>> stats.permtest_pearsonr(x, y)  # doctest: +SKIP
    (0.500040365781984, 0.000999000999000999)

    Also works with multiple columns by either broadcasting the smaller array
    to the larger:

    >>> z = x + np.random.normal(loc=1, size=100)
    >>> stats.permtest_pearsonr(x, np.column_stack([y, z]))
    (array([0.50004037, 0.25843187]), array([0.000999  , 0.01098901]))

    or by using matching columns in the two arrays (e.g., `x` and `y` vs
    `a` and `b`):

    >>> a, b = stats.make_correlated_xy(corr=0.9, size=100)
    >>> stats.permtest_pearsonr(np.column_stack([x, a]), np.column_stack([y, b]))
    (array([0.50004037, 0.89927523]), array([0.000999, 0.000999]))
    """  # noqa
    a, b, axis = _chk2_asarray(a, b, axis)
    rs = check_random_state(seed)

    if len(a) != len(b):
        raise ValueError('Provided arrays do not have same length')

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    if resamples is not None:
        if n_perm > resamples.shape[-1]:
            raise ValueError('Number of permutations requested exceeds size '
                             'of resampling array.')

    # divide by one forces coercion to float if ndim = 0
    true_corr = efficient_pearsonr(a, b)[0] / 1
    abs_true = np.abs(true_corr)

    permutations = np.ones(true_corr.shape)
    for perm in range(n_perm):
        # permute `a` and determine whether correlations exceed original
        if resamples is None:
            ap = a[rs.permutation(len(a))]
        else:
            ap = a[resamples[:, perm]]
        permutations += np.abs(efficient_pearsonr(ap, b)[0]) >= abs_true

    pvals = permutations / (n_perm + 1)  # + 1 in denom accounts for true_corr

    return true_corr, pvals


def _sw_nest_enrich_score(L, network_ind, one_sided=True):
    """
    Calculate the network enrichment score for a given statistic.

    Parameters
    ----------
    L : array_like, shape (n_vertices,)
        Statistics
    network_ind : array_like, shape (n_vertices,)
        Network indicator, where 1 indicates membership in the network of
        interest and 0 otherwise.
    one_sided : bool, optional
        Whether to perform a one-sided test. Default: True

    Returns
    -------
    enrich_score : float
        Network enrichment score
    """
    L_order = np.argsort(L)[::-1]
    L_sorted = L[L_order]
    network_ind_sorted = network_ind[L_order]

    if one_sided:
        L_sorted_abs = np.abs(L_sorted)
        P_hit_numerator = np.cumsum(L_sorted_abs * network_ind_sorted)
    else:
        P_hit_numerator = np.cumsum(network_ind_sorted)
    P_hit = P_hit_numerator / P_hit_numerator[-1]

    P_miss_numerator = np.cumsum(1 - network_ind_sorted)
    P_miss = P_miss_numerator / P_miss_numerator[-1]

    running_sum = P_hit - P_miss
    enrich_score = np.max(np.abs(running_sum))

    return enrich_score


def sw_nest(stat_emp, stat_perm, network_ind, one_sided=True):
    """
    Network Enrichment Significance Testing (NEST) from Weinstein et al., 2024.

    Check `original implementation <https://github.com/smweinst/NEST>`_ for more
    details.

    Parameters
    ----------
    stat_emp : array_like, shape (n_vertices,)
        Empirical statistics
    stat_perm : array_like, shape (n_permutations, n_vertices)
        Permuted statistics. Each row corresponds to a permutation calculated by
        permuting the subjects and re-estimating the statistic.
    network_ind : array_like, shape (n_vertices,)
        Network indicator, where 1 indicates membership in the network of
        interest and 0 otherwise.
    one_sided : bool, optional
        Whether to perform a one-sided test. Default: True

    Returns
    -------
    p : float
        Significance level

    References
    ----------
    .. [1] Weinstein, S. M., Vandekar, S. N., Li, B., Alexander-Bloch, A. F.,
       Raznahan, A., Li, M., Gur, R. E., Gur, R. C., Roalf, D. R., Park, M. T.
       M., Chakravarty, M., Baller, E. B., Linn, K. A., Satterthwaite, T. D., &
       Shinohara, R. T. (2024). Network enrichment significance testing in
       brain-phenotype association studies. Human Brain Mapping, 45(8), e26714.
       https://doi.org/10.1002/hbm.26714

    See Also
    --------
    netneurotools.stats.sw_nest_perm_ols
    """
    es_emp = _sw_nest_enrich_score(stat_emp, network_ind, one_sided=one_sided)
    es_perm = np.array([
        _sw_nest_enrich_score(s, network_ind, one_sided=one_sided) for s in stat_perm
    ])
    return (1 + np.sum(es_perm > es_emp)) / (1 + len(es_perm))


def sw_nest_perm_ols(
    observed_vars,  # (N, P)
    predictor_vars,  # (N,) or (N, 1)
    covariate_vars=None,  # (N,) or (N, C)
    freedman_lane=False,
    n_perm=1000,
    rng=None
):
    """
    Network Enrichment Significance Testing (NEST) from Weinstein et al., 2024.

    This function implements the permutation test for OLS from the NEST paper.
    Note that it does not generate the network enrichment score, but rather
    returns the empirical and permuted statistics for use in the `sw_nest` function.

    Check `original implementation <https://github.com/smweinst/NEST>`_ for more
    details.

    Parameters
    ----------
    observed_vars : array_like, shape (n_subjects, n_vertices)
        Observed variables
    predictor_vars : array_like, shape (n_subjects,)
        Predictor variable
    covariate_vars : array_like, shape (n_subjects, n_covars), optional
        Covariate variables. Default: None
    freedman_lane : bool, optional
        Whether to use the Freedman-Lane method. Default: False
    n_perm : int, optional
        Number of permutations to assess. Default: 1000
    rng : {int, np.random.Generator, np.random.RandomState}, optional
        Random number generator. Default: None

    Returns
    -------
    empirical : array_like, shape (n_vertices,)
        Empirical statistics
    permuted : array_like, shape (n_permutations, n_vertices)
        Permuted statistics. Each row corresponds to a permutation calculated by
        permuting the subjects and re-estimating the statistic.

    References
    ----------
    .. [1] Weinstein, S. M., Vandekar, S. N., Li, B., Alexander-Bloch, A. F.,
       Raznahan, A., Li, M., Gur, R. E., Gur, R. C., Roalf, D. R., Park, M. T.
       M., Chakravarty, M., Baller, E. B., Linn, K. A., Satterthwaite, T. D., &
       Shinohara, R. T. (2024). Network enrichment significance testing in
       brain-phenotype association studies. Human Brain Mapping, 45(8), e26714.
       https://doi.org/10.1002/hbm.26714

    See Also
    --------
    netneurotools.stats.sw_nest
    """
    if not _has_statsmodels:
        raise ImportError("statsmodels is required for this function")

    if not isinstance(rng, np.random.Generator):
        try:
            rng = np.random.default_rng(rng)
        except TypeError as e:
            raise TypeError(
                f"Cannnot initiate Random Generator from {rng}"
                ) from e

    if covariate_vars is None:
        covariate_vars = np.ones_like(predictor_vars)
    if predictor_vars.ndim == 1:
        predictor_vars = predictor_vars.reshape((-1, 1))
    if covariate_vars.ndim == 1:
        covariate_vars = covariate_vars.reshape((-1, 1))

    n, _ = observed_vars.shape
    perm_indices = np.array([rng.permutation(n) for _ in range(n_perm)])  # (n_perm, n)

    def _single_ols(_obs, _pred, metric=None):
        if metric is None:
            # assumes single predictor and appended intercept
            return sma.OLS(_obs, _pred).fit()
        elif metric == "coeff":
            return sma.OLS(_obs, _pred).fit().params[0, :]
        else:
            raise ValueError(f"{metric} is not valid")

    if not freedman_lane:
        _pred = sma.add_constant(
            np.c_[predictor_vars, covariate_vars],
            prepend=False, has_constant="skip"
        )
        # observed_vars ~ predictor_vars + covariate_vars + 1
        empirical = _single_ols(observed_vars, _pred, metric="coeff")

        permuted = []
        for i_perm in range(n_perm):
            _pred = sma.add_constant(
                np.c_[predictor_vars[perm_indices[i_perm], :], covariate_vars],
                prepend=False, has_constant="skip"
            )
            permuted.append(
                _single_ols(observed_vars, _pred, metric="coeff")
            )
        permuted = np.array(permuted)
    else:
        #
        _pred = sma.add_constant(
            np.c_[predictor_vars, covariate_vars],
            prepend=False, has_constant="skip"
        )
        _pred_cov = sma.add_constant(
            covariate_vars,
            prepend=False, has_constant="skip"
        )
        # observed_vars ~ predictor_vars + covariate_vars + 1
        empirical = _single_ols(observed_vars, _pred, metric="coeff")

        # observed_vars ~ covariate_vars + 1 -> residual
        ols_reduced = _single_ols(observed_vars, _pred_cov)
        resid = ols_reduced.resid

        permuted = []
        for i_perm in range(n_perm):
            # observed_vars = fitted values from ols_reduced + permuted residuals
            _obs_perm = ols_reduced.predict(_pred_cov) + resid[perm_indices[i_perm], :]
            # _obs_perm ~ predictor_vars + covariate_vars + 1
            permuted.append(
                _single_ols(_obs_perm, _pred, metric="coeff")
            )
        permuted = np.array(permuted)

    return empirical, permuted


def sw_spice(X, Y, n_perm=10000, rng=None):
    """
    Simple Permutation-based Intermodal Correspondence (SPICE) from Weinstein et al., 2021.

    Check `original implementation <https://github.com/smweinst/spice_test>`_ for more details.

    Parameters
    ----------
    X : array_like, shape (n_subjects, n_features)
        Data matrix for the first modality
    Y : array_like, shape (n_subjects, n_features)
        Data matrix for the second modality
    n_perm : int, optional
        Number of permutations to assess. Default: 10000
    rng : {int, np.random.Generator, np.random.RandomState}, optional
        Random number generator. Default: None

    Returns
    -------
    p : float
        Significance level

    References
    ----------
    .. [1] Weinstein, S. M., Vandekar, S. N., Adebimpe, A., Tapera, T. M.,
        Robert‐Fitzgerald, T., Gur, R. C., ... & Shinohara, R. T. (2021). A
        simple permutation‐based test of intermodal correspondence. Human brain
        mapping, 42(16), 5175-5187.
    """  # noqa E501
    if not isinstance(rng, np.random.Generator):
        try:
            rng = np.random.default_rng(rng)
        except TypeError as e:
            raise TypeError(
                f"Cannnot initiate Random Generator from {rng}"
                ) from e

    if X.shape != Y.shape:
        raise ValueError("X and Y should be the same shape!")
    n_subj, n_feat = X.shape  # noqa: F841

    stat_emp = np.mean([
        sstats.pearsonr(X[i, :], Y[i, :])[0]
        for i in range(n_subj)
    ])

    stat_perm = np.empty((n_perm,))
    for p in range(n_perm):
        Y_perm = rng.permutation(Y, axis=0)
        stat_perm[p] = np.mean([
            sstats.pearsonr(X[i, :], Y_perm[i, :])[0]
            for i in range(n_subj)
        ])
    return (np.sum(np.abs(stat_emp) < np.abs(stat_perm)) + 1) / (n_perm + 1)
