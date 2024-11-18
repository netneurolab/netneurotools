"""Functions for calculating regression."""

from itertools import combinations

import numpy as np
from tqdm import tqdm
import scipy.stats as sstats
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array


def _add_constant(data):
    """
    Add a constant (i.e., intercept) term to `data`.

    Parameters
    ----------
    data : (N, M) array_like
        Samples by features data array

    Returns
    -------
    data : (N, F) np.ndarray
        Where `F` is `M + 1`

    Examples
    --------
    >>> from netneurotools import stats

    >>> A = np.zeros((5, 5))
    >>> Ac = stats._add_constant(A)
    >>> Ac
    array([[0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.]])
    """
    data = check_array(data, ensure_2d=False)
    return np.column_stack([data, np.ones(len(data))])


def residualize(X, Y, Xc=None, Yc=None, normalize=True, add_intercept=True):
    """
    Return residuals of regression equation from `Y ~ X`.

    Parameters
    ----------
    X : (N[, R]) array_like
        Coefficient matrix of `R` variables for `N` subjects
    Y : (N[, F]) array_like
        Dependent variable matrix of `F` variables for `N` subjects
    Xc : (M[, R]) array_like, optional
        Coefficient matrix of `R` variables for `M` subjects. If not specified
        then `X` is used to estimate betas. Default: None
    Yc : (M[, F]) array_like, optional
        Dependent variable matrix of `F` variables for `M` subjects. If not
        specified then `Y` is used to estimate betas. Default: None
    normalize : bool, optional
        Whether to normalize (i.e., z-score) residuals. Will use residuals from
        `Yc ~ Xc` for generating mean and variance. Default: True
    add_intercept : bool, optional
        Whether to add intercept to `X` (and `Xc`, if provided). The intercept
        will not be removed, just used in beta estimation. Default: True

    Returns
    -------
    Yr : (N, F) numpy.ndarray
        Residuals of `Y ~ X`

    Notes
    -----
    If both `Xc` and `Yc` are provided, these are used to calculate betas which
    are then applied to `X` and `Y`.
    """
    if ((Yc is None and Xc is not None) or (Yc is not None and Xc is None)):
        raise ValueError('If processing against a comparative group, you must '
                         'provide both `Xc` and `Yc`.')

    X, Y = np.asarray(X), np.asarray(Y)

    if Yc is None:
        Xc, Yc = X.copy(), Y.copy()
    else:
        Xc, Yc = np.asarray(Xc), np.asarray(Yc)

    # add intercept to regressors if requested and calculate fit
    if add_intercept:
        X, Xc = _add_constant(X), _add_constant(Xc)
    betas, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)

    # remove intercept from regressors and betas for calculation of residuals
    if add_intercept:
        betas = betas[:-1]
        X, Xc = X[:, :-1], Xc[:, :-1]

    # calculate residuals
    Yr = Y - (X @ betas)
    Ycr = Yc - (Xc @ betas)

    if normalize:
        Yr = sstats.zmap(Yr, compare=Ycr)

    return Yr


def get_dominance_stats(X, y, use_adjusted_r_sq=True, verbose=False, n_jobs=1):
    """
    Return the dominance analysis statistics for multilinear regression.

    This is a rewritten & simplified version of [DA1]_. It is briefly
    tested against the original package, but still in early stages.
    Please feel free to report any bugs.

    Warning: Still work-in-progress. Parameters might change!

    Parameters
    ----------
    X : (N, M) array_like
        Input data
    y : (N,) array_like
        Target values
    use_adjusted_r_sq : bool, optional
        Whether to use adjusted r squares. Default: True
    verbose : bool, optional
        Whether to print debug messages. Default: False
    n_jobs : int, optional
        The number of jobs to run in parallel. Default: 1

    Returns
    -------
    model_metrics : dict
        The dominance metrics, currently containing `individual_dominance`,
        `partial_dominance`, `total_dominance`, and `full_r_sq`.
    model_r_sq : dict
        Contains all model r squares

    Notes
    -----
    Example usage

    .. code:: python

        from netneurotools.stats import get_dominance_stats
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)
        model_metrics, model_r_sq = get_dominance_stats(X, y)

    To compare with [DA1]_, use `use_adjusted_r_sq=False`

    .. code:: python

        from dominance_analysis import Dominance_Datasets
        from dominance_analysis import Dominance
        boston_dataset=Dominance_Datasets.get_boston()
        dominance_regression=Dominance(data=boston_dataset,
                                       target='House_Price',objective=1)
        incr_variable_rsquare=dominance_regression.incremental_rsquare()
        dominance_regression.dominance_stats()

    References
    ----------
    .. [DA1] https://github.com/dominance-analysis/dominance-analysis

    """
    # this helps to remove one element from a tuple
    def remove_ret(tpl, elem):
        lst = list(tpl)
        lst.remove(elem)
        return tuple(lst)

    # sklearn linear regression wrapper
    def get_reg_r_sq(X, y, use_adjusted_r_sq=True):
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        yhat = lin_reg.predict(X)
        SS_Residual = sum((y - yhat) ** 2)
        SS_Total = sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (float(SS_Residual)) / SS_Total
        adjusted_r_squared = 1 - (1 - r_squared) * \
            (len(y) - 1) / (len(y) - X.shape[1] - 1)
        if use_adjusted_r_sq:
            return adjusted_r_squared
        else:
            return r_squared

    # helper function to compute r_sq for a given idx_tuple
    def compute_r_sq(idx_tuple):
        return idx_tuple, get_reg_r_sq(X[:, idx_tuple],
                                       y,
                                       use_adjusted_r_sq=use_adjusted_r_sq)

    # generate all predictor combinations in list (num of predictors) of lists
    n_predictor = X.shape[-1]
    # n_comb_len_group = n_predictor - 1
    predictor_combs = [list(combinations(range(n_predictor), i))
                       for i in range(1, n_predictor + 1)]
    if verbose:
        print(f"[Dominance analysis] Generated \
              {len([v for i in predictor_combs for v in i])} combinations")

    model_r_sq = dict()
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_r_sq)(idx_tuple)
        for len_group in tqdm(predictor_combs,
                              desc='num-of-predictor loop',
                              disable=not verbose)
        for idx_tuple in tqdm(len_group,
                              desc='insider loop',
                              disable=not verbose))

    # extract r_sq from results
    for idx_tuple, r_sq in results:
        model_r_sq[idx_tuple] = r_sq

    if verbose:
        print(f"[Dominance analysis] Acquired {len(model_r_sq)} r^2's")

    # getting all model metrics
    model_metrics = dict([])

    # individual dominance
    individual_dominance = []
    for i_pred in range(n_predictor):
        individual_dominance.append(model_r_sq[(i_pred,)])
    individual_dominance = np.array(individual_dominance).reshape(1, -1)
    model_metrics["individual_dominance"] = individual_dominance

    # partial dominance
    partial_dominance = [[] for _ in range(n_predictor - 1)]
    for i_len in range(n_predictor - 1):
        i_len_combs = list(combinations(range(n_predictor), i_len + 2))
        for j_node in range(n_predictor):
            j_node_sel = [v for v in i_len_combs if j_node in v]
            reduced_list = [remove_ret(comb, j_node) for comb in j_node_sel]
            diff_values = [
                model_r_sq[j_node_sel[i]] - model_r_sq[reduced_list[i]]
                for i in range(len(reduced_list))]
            partial_dominance[i_len].append(np.mean(diff_values))

    # save partial dominance
    partial_dominance = np.array(partial_dominance)
    model_metrics["partial_dominance"] = partial_dominance
    # get total dominance
    total_dominance = np.mean(
        np.r_[individual_dominance, partial_dominance], axis=0)
    # test and save total dominance
    assert np.allclose(total_dominance.sum(),
                       model_r_sq[tuple(range(n_predictor))]), \
           "Sum of total dominance is not equal to full r square!"
    model_metrics["total_dominance"] = total_dominance
    # save full r^2
    model_metrics["full_r_sq"] = model_r_sq[tuple(range(n_predictor))]

    return model_metrics, model_r_sq
