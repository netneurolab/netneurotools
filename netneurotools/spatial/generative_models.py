"""Functions for generating brain maps."""

import warnings
import numpy as np
from .spatial_stats import morans_i, _I_ev, _I_var
from tqdm import tqdm, trange

from .. import has_numba
if has_numba:
    from numba import njit


def generate_ac_maps(X, weights, I_trg=None, n_maps=10, epsilon=0.0001,
                     standardized=True, seed=None, **kwargs):
    """
    Generate surrogate maps preserving the autocorrelation structure of `X`.

    This function calculates the Moran's I of X with respect to each weight matrix
    provided in `weights`. Then, starting from Y, a random permutation of `X`, random
    pairs of values are swapped until the Moran's I of Y matches the Moran's I values
    of `X` within the specified tolerance. Alternative target values of Moran's I can
    also be specified using the `I_trg` parameter.

    Parameters
    ----------
    X: array-like of shape (n,)
        Vector of empirical values to be randomized.
    weights: array-like of shape (n, n) or (k, n, n)
        Weight matrix or collection of weight matrices used to compute Moran's I. Each
        matrix captures a unique type of pairwise interactions between brain regions.
        When multiple weight matrices are provided, the optimization matches the target
        Moran's I values for all of them simultaneously.
    I_trg: float or array-like of shape (k,), optional
        Target Moran's I value(s). If not provided, the Moran's I of `X` is
        computed for each weight matrix and used as the target.
    n_maps : int, default: 10
        Number of surrogate maps to generate.
    epsilon: float or array-like of shape (m,), default: 0.0001
        Error tolerance for the optimization. If an array is provided, the algorithm
        records the first solution that reaches each tolerance level.
    standardized: bool, default: True
        Whether Moran's I values are standardized or not.
    seed : int, optional
        Seed used to initialize the random number generator.
    **kwargs
        Additional keyword arguments passed to :func:`generate_ac_map`.

    Returns
    -------
    Y_all : (n,) or (n_maps, n) or (n_maps, n_eps, n) ndarray
        Surrogate maps whose autocorrelation structure matches the autocorrelation
        structure of `X` up to the tolerance value(s) specified in `epsilon`.
    """
    rng = np.random.default_rng(seed)

    epsilon = np.atleast_1d(epsilon)
    n_eps = len(epsilon)

    # Create results array
    Y_all = np.zeros((n_maps, n_eps, len(X)))

    # Convert weights into array of shape (k,n,n)
    weights = _ensure_weight_array(weights)

    # Calculate moran's I values associated with the empirical map to simulate
    if I_trg is None:
        I_trg = [morans_i(X, w, standardized=standardized) for w in weights]

    # Specify the maximum number of attempts (after which we exit the while loop)
    max_attempts = n_maps * 10

    i = 0
    attempts = 0
    with tqdm(total=n_maps) as pbar:
        while i < n_maps:
            attempts += 1

            # Instantiate random vector
            Y_random = rng.permutation(X)

            # Simulate map with preserved autocorrelation structure
            Y, _, _, error = permute_ac_map(
                Y_random, weights, I_trg, epsilon=epsilon, standardized=standardized,
                rng=rng, **kwargs)

            # Only store results if simulation reached the epsilon value
            if error < epsilon[-1]:
                Y_all[i] = Y
                i += 1
                pbar.update(1)

            if attempts > max_attempts:
                warnings.warn(f"Only generated {i}/{n_maps} maps after"
                              f" {attempts} attempts.", stacklevel=2)
                return np.squeeze(Y_all)

    return np.squeeze(Y_all)


def permute_ac_map(Y, weights, I_trg, epsilon=0.0001, temp=1, niter=100, frac=0.5,
                   max_stage=100, rng=None, seed=None, verbose=False,
                   standardized=True, use_numba=False):
    """
    Permute a random map `Y` until its autocorrelation structure is equal to `I_trg`.

    This function uses a simulated annealing procedure to permute pair of values in
    a random map `Y` until its autocorrelation structure with respect to the weight
    matrices in `weights` matches the autocorrelation structure specified in `I_trg`.

    Parameters
    ----------
    Y : (n,) ndarray
        Initial vector to optimize. This is typically a random permutation of
        some empirical data.
    weights: (n, n) or (k, n, n) ndarray
        Weight matrix or collection of weight matrices used to compute Moran's I. Each
        matrix captures a unique type of pairwise interaction between brain regions.
        When multiple weight matrices are provided, the optimization matches the target
        Moran's I values for all of them simultaneously.
    I_trg: float or array-like of shape (k,)
        Target Moran's I values for the generative algorithm. If multiple weight
        matrices are supplied, one target value must be provided for each matrix.
    epsilon: float or array-like
        Error tolerance for the optimization. If an array is provided, the algorithm
        records the first solution that reaches each tolerance level.
    temp : float, default: 1
        Initial temperature for the simulated annealing algorithm.
    niter : int, default: 100
        Number of swap proposals per annealing stage.
    frac : float, default: 0.5
        Multiplicative cooling factor applied to the temperature after each
        annealing stage.
    max_stage: int, default: 100
        Maximum number of stages in the optimization schedule.
    rng : numpy.random.Generator, optional
        Random number generator used to perform random swaps during the optimization.
        If `None` (default), a new generator is created from `seed`.
    seed : int, optional
        Seed used to initialize the random number generator. Ignored if `rng` is
        provided.
    verbose : bool or int, default: False
        Levels of verbose during the optimization procedure.
    standardized: bool, default: True
        Whether Moran's I values are standardized or not.
    use_numba : bool, optional
        Whether to use numba for calculation. Default: False (if numba is
        installed).

    Returns
    -------
    Y : (n,) ndarray or list of ndarray
        Simulated maps with a specified autocorrelation structure (for each tolerance
        value).
    it : int or list of int
        Iteration number(s) at which the solution is recorded (for each tolerance
        value).
    temp : float or list of float
        Annealing temperature(s) at which the solution is recorded (for each tolerance
        value).
    error : float or list of float
        Final optimization error(s) (for each tolerance value).
    """
    # Seed the random number generator
    if rng is None:
        rng = np.random.default_rng(seed)

    # Check if numba is installed
    if use_numba:
        if not has_numba:
            raise ValueError("Numba not installed; cannot use numba for calculation")

    # Convert weights and I_trg into arrays of shape (k,n,n) and (k) respectively
    weights = _ensure_weight_array(weights)
    I_trg = np.atleast_1d(I_trg)

    # Evaluate whether the matrix is symmetric (to speed-up calculations)
    is_symmetric = all(np.allclose(w, w.T) for w in weights)

    # Pre-compute constants for calculating Moran's I
    k, n_nodes, _ = weights.shape
    z = Y - Y.mean()
    den = (z * z).sum()
    const = n_nodes / weights.sum(axis=(1, 2)) / den

    if standardized:
        I_ev = np.array([_I_ev(w) for w in weights])
        I_std = np.array([np.sqrt(_I_var(w)) for w in weights])
    else:
        I_ev = np.zeros(k)
        I_std = np.ones(k)

    # Initialize new values vector
    new_Y = Y.copy()
    new_z = z.copy()

    # Compute Wz and WTz matrices
    Wz = weights @ z
    if is_symmetric:
        WTz = weights.transpose(0, 2, 1) @ z
    else:
        WTz = None

    # Compute initial Moran's I (non-standardized), Wz matrix and error (standardized)
    I_surr = const * np.sum(z * Wz, axis=1)
    error = _standardized_error(I_surr, I_trg, I_ev, I_std)

    Y_all, it_all, temp_all, error_all = [], [], [], []

    # Setup epsilon values
    epsilon = np.atleast_1d(epsilon)
    if np.any(np.diff(epsilon) < 0):
        raise ValueError("epsilon must be sorted in increasing order.")
    n_eps = len(epsilon)
    curr_eps = epsilon[0]
    curr_eps_nb = 0

    it = 0
    for istage in trange(max_stage) if verbose > 0 else range(max_stage):
        naccept = 0
        nrand = 0

        if curr_eps_nb == n_eps:
            break
        if error < curr_eps:
            _record_state(Y_all, it_all, temp_all, error_all,
                          new_Y, it, temp, error)
            curr_eps_nb, curr_eps, done = _advance_epsilon(curr_eps_nb, epsilon, n_eps)
            if done:
                break

        for _ in trange(niter) if verbose > 2 else range(niter):
            it += 1

            # Do a random swap then compute updated Moran's I + error
            n1 = rng.integers(n_nodes)
            while True:
                n2 = rng.integers(n_nodes)
                if n2 != n1:
                    break

            if use_numba:
                I_new = _update_multi_morans_i_numba(
                    weights, new_z, Wz, WTz, const, n1, n2, I_surr, is_symmetric)
                error_new = _standardized_error_numba(I_new, I_trg, I_ev, I_std)
            else:
                I_new = _update_multi_morans_i(
                    weights, new_z, Wz, WTz, const, n1, n2, I_surr, is_symmetric)
                error_new = _standardized_error(I_new, I_trg, I_ev, I_std)

            if _accept_swap(error_new, error, temp, rng):

                naccept += 1
                if error_new >= error:
                    nrand += 1

                error = error_new
                I_surr = I_new

                delta = new_z[n2] - new_z[n1]
                if use_numba:
                    _update_Wz_numba(Wz, WTz, delta, weights, n1, n2, is_symmetric)
                else:
                    Wz, WTz = _update_Wz(Wz, WTz, delta, weights, n1, n2, is_symmetric)
                new_Y[n1], new_Y[n2] = new_Y[n2], new_Y[n1]
                new_z[n1], new_z[n2] = new_z[n2], new_z[n1]

                if error < curr_eps:
                    _record_state(Y_all, it_all, temp_all, error_all,
                                  new_Y, it, temp, error)
                    curr_eps_nb, curr_eps, done = _advance_epsilon(curr_eps_nb, epsilon,
                                                                   n_eps)
                    if done:
                        break

        temp *= frac
        if verbose > 1:
            _log_stage_info(istage, temp, error, naccept, niter, nrand)

    # We're at the final iteration number without reaching the optimal eps
    if (error >= curr_eps) and (curr_eps_nb < n_eps):
        _record_state(Y_all, it_all, temp_all, error_all,
                      new_Y, it, temp, error)

    if n_eps == 1:
        return Y_all[0], it_all[0], temp_all[0], error_all[0]
    else:
        return Y_all, it_all, temp_all, error_all


def _update_Wz(Wz, WTz, delta, weights, n1, n2, is_symmetric):
    Wz += delta * ((weights[:, :, n1] - weights[:, :, n2]))
    if not is_symmetric:
        WTz += delta * ((weights[:, n1, :] - weights[:, n2, :]))
    return Wz, WTz


def _update_Wz_numba(Wz, WTz, delta, weights, n1, n2, is_symmetric):
    k, n_nodes = Wz.shape
    for a in range(k):
        for m in range(n_nodes):
            Wz[a, m] += delta * (
                weights[a, m, n1] - weights[a, m, n2]
            )
    if not is_symmetric:
        for a in range(k):
            for m in range(n_nodes):
                WTz[a, m] += delta * (
                    weights[a, n1, m] - weights[a, n2, m]
                )


def _standardized_error(I_curr, I_trg, I_mean, I_std):

    error = ((I_curr - I_mean) / I_std) - I_trg

    return np.max(np.abs(error))


def _standardized_error_numba(I_curr, I_trg, I_mean, I_std):

    err = 0.0

    for i in range(I_curr.shape[0]):
        e = abs(((I_curr[i] - I_mean[i]) / I_std[i]) - I_trg[i])
        if e > err:
            err = e

    return err


def _update_multi_morans_i(w, z, Wz, WTz, const, i, j, original_I, is_symmetric):

    delta = z[j] - z[i]

    if is_symmetric:
        delta_I = 2 * const * delta * ((Wz[:, i] - Wz[:, j]) - delta * w[:, j, i])
    else:
        delta_I = const * delta * (
            (Wz[:, i] - Wz[:, j]) + (WTz[:, i] - WTz[:, j]) -
             delta * (w[:, i, j] + w[:, j, i]))

    I_new = original_I + delta_I

    return I_new


def _update_multi_morans_i_numba(w, z, Wz, WTz, const, i, j, original_I, is_symmetric):

    delta = z[j] - z[i]

    k = w.shape[0]
    I_new = np.empty(k)

    for a in range(k):
        if is_symmetric:
            delta_I = (2 * const[a] * delta *
                ((Wz[a, i] - Wz[a, j]) -
                 delta * w[a, j, i]))
        else:
            delta_I = (const[a] * delta *
                ((Wz[a, i] - Wz[a, j]) + (WTz[a, i] - WTz[a, j]) -
                 delta * (w[a, i, j] + w[a, j, i])))

        I_new[a] = original_I[a] + delta_I

    return I_new


if has_numba:
    _update_Wz_numba = njit(_update_Wz_numba)
    _standardized_error_numba = njit(_standardized_error_numba)
    _update_multi_morans_i_numba = njit(_update_multi_morans_i_numba)


def _accept_swap(error_new, error, temp, rng):
    return error_new < error or rng.random() < np.exp(-(error_new - error) / temp)


def _log_stage_info(stage, temp, error, naccept, niter, nrand):
    print(f"\nstage {stage}, temp {temp:.5f}, best energy {error:.6f}, "
          f"frac of accepted moves: {naccept / niter:.3f}")
    if naccept > 0:
        print(f"frac of random moves: {nrand / naccept:.3f}")


def _record_state(Y_all, it_all, temp_all, error_all, new_Y, it, temp, error):
    Y_all.append(new_Y.copy())
    it_all.append(it)
    temp_all.append(temp)
    error_all.append(error)


def _advance_epsilon(curr_eps_nb, epsilon, n_eps):
    curr_eps_nb += 1
    if curr_eps_nb == n_eps:
        return curr_eps_nb, epsilon[curr_eps_nb - 1], True
    else:
        return curr_eps_nb, epsilon[curr_eps_nb], False


def _ensure_weight_array(weights):

    weights = np.asarray(weights, dtype=float)

    # Convert (n, n) to (1, n, n)
    if weights.ndim == 2:
        weights = weights[np.newaxis, ...]
    elif weights.ndim != 3:
        raise ValueError("`weights` must have shape (n, n) or (k, n, n).")

    # Ensure that weight matrices are square
    k, n, m = weights.shape
    if n != m:
        raise ValueError("Weight matrices must be square.")

    return weights
