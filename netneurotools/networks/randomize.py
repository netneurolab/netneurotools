"""Functions for generating randomized networks."""

import bct
import numpy as np
from tqdm import tqdm
from sklearn.utils.validation import check_random_state

try:
    from numba import njit

    has_numba = True
except ImportError:
    has_numba = False


def randmio_und(W, itr):
    """
    Optimized version of randmio_und.

    This function randomizes an undirected network, while preserving the
    degree distribution. The function does not preserve the strength
    distribution in weighted networks.

    This function is significantly faster if numba is enabled, because
    the main overhead is `np.random.randint`, see `here <https://stackoverflow.com/questions/58124646/why-in-python-is-random-randint-so-much-slower-than-random-random>`_

    Parameters
    ----------
    W : (N, N) array-like
        Undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.

    Returns
    -------
    W : (N, N) array-like
        Randomized network
    eff : int
        number of actual rewirings carried out
    """  # noqa: E501
    W = W.copy()
    n = len(W)
    i, j = np.where(np.triu(W > 0, 1))
    k = len(i)
    itr *= k

    # maximum number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))
    # actual number of successful rewirings
    eff = 0

    for _ in range(int(itr)):
        att = 0
        while att <= max_attempts:  # while not rewired
            while True:
                e1, e2 = np.random.randint(k), np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a, b = i[e1], j[e1]
                c, d = i[e2], j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            # flip edge c-d with 50% probability
            # to explore all potential rewirings
            if np.random.random() > 0.5:
                i[e2], j[e2] = d, c
                c, d = d, c

            # rewiring condition
            # not flipped
            # a--b    a  b
            #      TO  X
            # c--d    c  d
            # if flipped
            # a--b    a--b    a  b
            #      TO      TO  X
            # c--d    d--c    d  c
            if not (W[a, d] or W[c, b]):
                W[a, d] = W[a, b]
                W[a, b] = 0
                W[d, a] = W[b, a]
                W[b, a] = 0
                W[c, b] = W[c, d]
                W[c, d] = 0
                W[b, c] = W[d, c]
                W[d, c] = 0

                j[e1] = d
                j[e2] = b  # reassign edge indices
                eff += 1
                break
            att += 1

    return W, eff


if has_numba:
    randmio_und = njit(randmio_und)


def match_length_degree_distribution(
    W, D, nbins=10, nswap=1000, replacement=False, weighted=True, seed=None
):
    """
    Generate degree- and edge length-preserving surrogate connectomes.

    Parameters
    ----------
    W : (N, N) array-like
        weighted or binary symmetric connectivity matrix.
    D : (N, N) array-like
        symmetric distance matrix.
    nbins : int
        number of distance bins (edge length matrix is performed by swapping connections
        in the same bin). Default = 10.
    nswap : int
        total number of edge swaps to perform. Recommended = nnodes * 20.
        Default = 1000.
    replacement : bool, optional
        if True all the edges are available for swapping. Default = False.
    weighted : bool, optional
        if True the function returns a weighted matrix. Default = True.
    seed : float, optional
        Random seed. Default = None

    Returns
    -------
    newB : (N, N) array-like
        binary rewired matrix
    newW: (N, N) array-like
        weighted rewired matrix. Returns matrix of zeros if weighted=False.
    nr : int
        number of successful rewires

    Notes
    -----
    Takes a weighted, symmetric connectivity matrix `data` and Euclidean/fiber
    length matrix `distance` and generates a randomized network with:
        1. exactly the same degree sequence
        2. approximately the same edge length distribution
        3. exactly the same edge weight distribution
        4. approximately the same weight-length relationship

    Reference
    ---------
    Betzel, R. F., Bassett, D. S. (2018) Specificity and robustness of
    long-distance connections in weighted, interareal connectomes. PNAS.
    """
    rs = check_random_state(seed)
    N = len(W)
    # divide the distances by lengths
    bins = np.linspace(D[D.nonzero()].min(), D[D.nonzero()].max(), nbins + 1)
    bins[-1] += 1
    L = np.zeros((N, N))
    for n in range(nbins):
        i, j = np.where(np.logical_and(bins[n] <= D, D < bins[n + 1]))
        L[i, j] = n + 1

    # binarized connectivity
    B = (W != 0).astype(np.int_)

    # existing edges (only upper triangular cause it's symmetric)
    cn_x, cn_y = np.where(np.triu((B != 0) * B, k=1))

    tries = 0
    nr = 0
    newB = np.copy(B)

    while (len(cn_x) >= 2) & (nr < nswap):
        # choose randomly the edge to be rewired
        r = rs.randint(len(cn_x))
        n_x, n_y = cn_x[r], cn_y[r]
        tries += 1

        # options to rewire with
        # connected nodes that doesn't involve (n_x, n_y)
        index = (cn_x != n_x) & (cn_y != n_y) & (cn_y != n_x) & (cn_x != n_y)
        if len(np.where(index)[0]) == 0:
            cn_x = np.delete(cn_x, r)
            cn_y = np.delete(cn_y, r)

        else:
            ops1_x, ops1_y = cn_x[index], cn_y[index]
            # options that will preserve the distances
            # (ops1_x, ops1_y) such that
            # L(n_x,n_y) = L(n_x, ops1_x) & L(ops1_x,ops1_y) = L(n_y, ops1_y)
            index = (L[n_x, n_y] == L[n_x, ops1_x]) & (
                L[ops1_x, ops1_y] == L[n_y, ops1_y]
            )
            if len(np.where(index)[0]) == 0:
                cn_x = np.delete(cn_x, r)
                cn_y = np.delete(cn_y, r)

            else:
                ops2_x, ops2_y = ops1_x[index], ops1_y[index]
                # options of edges that didn't exist before
                index = [
                    (newB[min(n_x, ops2_x[i])][max(n_x, ops2_x[i])] == 0)
                    & (newB[min(n_y, ops2_y[i])][max(n_y, ops2_y[i])] == 0)
                    for i in range(len(ops2_x))
                ]
                if len(np.where(index)[0]) == 0:
                    cn_x = np.delete(cn_x, r)
                    cn_y = np.delete(cn_y, r)

                else:
                    ops3_x, ops3_y = ops2_x[index], ops2_y[index]

                    # choose randomly one edge from the final options
                    r1 = rs.randint(len(ops3_x))
                    nn_x, nn_y = ops3_x[r1], ops3_y[r1]

                    # Disconnect the existing edges
                    newB[n_x, n_y] = 0
                    newB[nn_x, nn_y] = 0
                    # Connect the new edges
                    newB[min(n_x, nn_x), max(n_x, nn_x)] = 1
                    newB[min(n_y, nn_y), max(n_y, nn_y)] = 1
                    # one successfull rewire!
                    nr += 1

                    # rewire with replacement
                    if replacement:
                        cn_x[r], cn_y[r] = min(n_x, nn_x), max(n_x, nn_x)
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x[index], cn_y[index] = min(n_y, nn_y), max(n_y, nn_y)
                    # rewire without replacement
                    else:
                        cn_x = np.delete(cn_x, r)
                        cn_y = np.delete(cn_y, r)
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x = np.delete(cn_x, index)
                        cn_y = np.delete(cn_y, index)

    if nr < nswap:
        print(f"I didn't finish, out of rewirable edges: {len(cn_x)}")

    i, j = np.triu_indices(N, k=1)
    # Make the connectivity matrix symmetric
    newB[j, i] = newB[i, j]

    # check the number of edges is preserved
    if len(np.where(B != 0)[0]) != len(np.where(newB != 0)[0]):
        print(
            f"ERROR --- number of edges changed, \
            B:{len(np.where(B != 0)[0])}, newB:{len(np.where(newB != 0)[0])}"
        )
    # check that the degree of the nodes it's the same
    for i in range(N):
        if np.sum(B[i]) != np.sum(newB[i]):
            print(
                f"ERROR --- node {i} changed k by: \
                {np.sum(B[i]) - np.sum(newB[i])}"
            )

    newW = np.zeros((N, N))
    if weighted:
        # Reassign the weights
        mask = np.triu(B != 0, k=1)
        inids = D[mask]
        iniws = W[mask]
        inids_index = np.argsort(inids)
        # Weights from the shortest to largest edges
        iniws = iniws[inids_index]
        mask = np.triu(newB != 0, k=1)
        finds = D[mask]
        i, j = np.where(mask)
        # Sort the new edges from the shortest to the largest
        finds_index = np.argsort(finds)
        i_sort = i[finds_index]
        j_sort = j[finds_index]
        # Assign the initial sorted weights
        newW[i_sort, j_sort] = iniws
        # Make it symmetrical
        newW[j_sort, i_sort] = iniws

    return newB, newW, nr


def strength_preserving_rand_sa(
    A,
    rewiring_iter=10,
    nstage=100,
    niter=10000,
    temp=1000,
    frac=0.5,
    energy_type="sse",
    energy_func=None,
    R=None,
    connected=None,
    verbose=False,
    seed=None,
):
    """
    Strength-preserving network randomization using simulated annealing.

    Randomize an undirected weighted network, while preserving
    the degree and strength sequences using simulated annealing.

    This function allows for a flexible choice of energy function.

    Parameters
    ----------
    A : (N, N) array-like
        Undirected weighted connectivity matrix
    rewiring_iter : int, optional
        Rewiring parameter. Default = 10.
        Each edge is rewired approximately rewiring_iter times.
    nstage : int, optional
        Number of annealing stages. Default = 100.
    niter : int, optional
        Number of iterations per stage. Default = 10000.
    temp : float, optional
        Initial temperature. Default = 1000.
    frac : float, optional
        Fractional decrease in temperature per stage. Default = 0.5.
    energy_type: str, optional
        Energy function to minimize. Can be either:
            'sse': Sum of squared errors between strength sequence vectors
                   of the original network and the randomized network
            'max': Maximum absolute error
            'mae': Mean absolute error
            'mse': Mean squared error
            'rmse': Root mean squared error
        Default = 'sse'.
    energy_func: callable, optional
        Callable with two positional arguments corresponding to
        two strength sequence numpy arrays that returns an energy value.
        Overwrites “energy_type”.
        See “energy_type” for specifying a predefined energy type instead.
    R : (N, N) array-like, optional
        Pre-randomized connectivity matrix.
        If None, a rewired connectivity matrix is generated using the
        Maslov & Sneppen algorithm.
        Default = None.
    connected: bool, optional
        Whether to ensure connectedness of the randomized network.
        By default, this is inferred from data.
    verbose: bool, optional
        Whether to print status to screen at the end of every stage.
        Default = False.
    seed: float, optional
        Random seed. Default = None.

    Returns
    -------
    B : (N, N) array-like
        Randomized connectivity matrix
    min_energy : float
        Minimum energy obtained by annealing

    Notes
    -----
    Uses Maslov & Sneppen rewiring model to produce a
    surrogate connectivity matrix, B, with the same
    size, density, and degree sequence as A.
    The weights are then permuted to optimize the
    match between the strength sequences of A and B
    using simulated annealing.

    This function is adapted from a function written in MATLAB
    by Richard Betzel.

    References
    ----------
    Misic, B. et al. (2015) Cooperative and Competitive Spreading Dynamics
    on the Human Connectome. Neuron.
    Milisav, F. et al. (2024) A simulated annealing algorithm for
    randomizing weighted networks.
    """
    try:
        A = np.asarray(A)
    except TypeError as err:
        msg = "A must be array_like. Received: {}.".format(type(A))
        raise TypeError(msg) from err

    if frac > 1 or frac <= 0:
        msg = "frac must be between 0 and 1. " "Received: {}.".format(frac)
        raise ValueError(msg)

    rs = check_random_state(seed)

    n = A.shape[0]
    s = np.sum(A, axis=1)  # strengths of A

    # Maslov & Sneppen rewiring
    if R is None:
        # ensuring connectedness if the original network is connected
        if connected is None:
            connected = False if bct.number_of_components(A) > 1 else True
        if connected:
            B = bct.randmio_und_connected(A, rewiring_iter, seed=seed)[0]
        else:
            B = bct.randmio_und(A, rewiring_iter, seed=seed)[0]
    else:
        B = R.copy()

    u, v = np.triu(B, k=1).nonzero()  # upper triangle indices
    wts = np.triu(B, k=1)[(u, v)]  # upper triangle values
    m = len(wts)
    sb = np.sum(B, axis=1)  # strengths of B

    if energy_func is not None:
        energy = energy_func(s, sb)
    elif energy_type == "sse":
        energy = np.sum((s - sb) ** 2)
    elif energy_type == "max":
        energy = np.max(np.abs(s - sb))
    elif energy_type == "mae":
        energy = np.mean(np.abs(s - sb))
    elif energy_type == "mse":
        energy = np.mean((s - sb) ** 2)
    elif energy_type == "rmse":
        energy = np.sqrt(np.mean((s - sb) ** 2))
    else:
        msg = (
            "energy_type must be one of 'sse', 'max', "
            "'mae', 'mse', or 'rmse'. Received: {}.".format(energy_type)
        )
        raise ValueError(msg)

    energymin = energy
    wtsmin = wts.copy()

    if verbose:
        print("\ninitial energy {:.5f}".format(energy))

    for istage in tqdm(range(nstage), desc="annealing progress"):
        naccept = 0
        for _ in range(niter):
            # permutation
            e1 = rs.randint(m)
            e2 = rs.randint(m)

            a, b = u[e1], v[e1]
            c, d = u[e2], v[e2]

            sb_prime = sb.copy()
            sb_prime[[a, b]] = sb_prime[[a, b]] - wts[e1] + wts[e2]
            sb_prime[[c, d]] = sb_prime[[c, d]] + wts[e1] - wts[e2]

            if energy_func is not None:
                energy_prime = energy_func(sb_prime, s)
            elif energy_type == "sse":
                energy_prime = np.sum((sb_prime - s) ** 2)
            elif energy_type == "max":
                energy_prime = np.max(np.abs(sb_prime - s))
            elif energy_type == "mae":
                energy_prime = np.mean(np.abs(sb_prime - s))
            elif energy_type == "mse":
                energy_prime = np.mean((sb_prime - s) ** 2)
            elif energy_type == "rmse":
                energy_prime = np.sqrt(np.mean((sb_prime - s) ** 2))
            else:
                msg = (
                    "energy_type must be one of 'sse', 'max', "
                    "'mae', 'mse', or 'rmse'. "
                    "Received: {}.".format(energy_type)
                )
                raise ValueError(msg)

            # permutation acceptance criterion
            if energy_prime < energy or rs.rand() < np.exp(
                -(energy_prime - energy) / temp
            ):
                sb = sb_prime.copy()
                wts[[e1, e2]] = wts[[e2, e1]]
                energy = energy_prime
                if energy < energymin:
                    energymin = energy
                    wtsmin = wts.copy()
                naccept = naccept + 1

        # temperature update
        temp = temp * frac
        if verbose:
            print(
                "\nstage {:d}, temp {:.5f}, best energy {:.5f}, "
                "frac of accepted moves {:.3f}".format(
                    istage, temp, energymin, naccept / niter
                )
            )

    B = np.zeros((n, n))
    B[(u, v)] = wtsmin
    B = B + B.T

    return B, energymin


def strength_preserving_rand_sa_mse_opt(
    A,
    rewiring_iter=10,
    nstage=100,
    niter=10000,
    temp=1000,
    frac=0.5,
    R=None,
    connected=None,
    verbose=False,
    seed=None,
):
    """
    Strength-preserving network randomization using simulated annealing.

    Randomize an undirected weighted network, while preserving
    the degree and strength sequences using simulated annealing.

    This function has been optimized for speed but only allows the
    mean squared error energy function.

    Parameters
    ----------
    A : (N, N) array-like
        Undirected weighted connectivity matrix
    rewiring_iter : int, optional
        Rewiring parameter. Default = 10.
        Each edge is rewired approximately rewiring_iter times.
    nstage : int, optional
        Number of annealing stages. Default = 100.
    niter : int, optional
        Number of iterations per stage. Default = 10000.
    temp : float, optional
        Initial temperature. Default = 1000.
    frac : float, optional
        Fractional decrease in temperature per stage. Default = 0.5.
    R : (N, N) array-like, optional
        Pre-randomized connectivity matrix.
        If None, a rewired connectivity matrix is generated using the
        Maslov & Sneppen algorithm.
        Default = None.
    connected: bool, optional
        Whether to ensure connectedness of the randomized network.
        By default, this is inferred from data.
    verbose: bool, optional
        Whether to print status to screen at the end of every stage.
        Default = False.
    seed: float, optional
        Random seed. Default = None.

    Returns
    -------
    B : (N, N) array-like
        Randomized connectivity matrix
    min_energy : float
        Minimum energy obtained by annealing

    Notes
    -----
    Uses Maslov & Sneppen rewiring model to produce a
    surrogate connectivity matrix, B, with the same
    size, density, and degree sequence as A.
    The weights are then permuted to optimize the
    match between the strength sequences of A and B
    using simulated annealing.

    This function is adapted from a function written in MATLAB
    by Richard Betzel and was optimized by Vincent Bazinet.

    References
    ----------
    Misic, B. et al. (2015) Cooperative and Competitive Spreading Dynamics
    on the Human Connectome. Neuron.
    Milisav, F. et al. (2024) A simulated annealing algorithm for
    randomizing weighted networks.
    """
    try:
        A = np.asarray(A)
    except TypeError as err:
        msg = "A must be array_like. Received: {}.".format(type(A))
        raise TypeError(msg) from err

    if frac > 1 or frac <= 0:
        msg = "frac must be between 0 and 1. " "Received: {}.".format(frac)
        raise ValueError(msg)

    rs = check_random_state(seed)

    n = A.shape[0]
    s = np.sum(A, axis=1)  # strengths of A

    # Maslov & Sneppen rewiring
    if R is None:
        # ensuring connectedness if the original network is connected
        if connected is None:
            connected = False if bct.number_of_components(A) > 1 else True
        if connected:
            B = bct.randmio_und_connected(A, rewiring_iter, seed=seed)[0]
        else:
            B = bct.randmio_und(A, rewiring_iter, seed=seed)[0]
    else:
        B = R.copy()

    u, v = np.triu(B, k=1).nonzero()  # upper triangle indices
    wts = np.triu(B, k=1)[(u, v)]  # upper triangle values
    m = len(wts)
    sb = np.sum(B, axis=1)  # strengths of B

    energy = np.mean((s - sb) ** 2)

    energymin = energy
    wtsmin = wts.copy()

    if verbose:
        print("\ninitial energy {:.5f}".format(energy))

    for istage in tqdm(range(nstage), desc="annealing progress"):
        naccept = 0
        for (e1, e2), prob in zip(rs.randint(m, size=(niter, 2)), rs.rand(niter)):
            # permutation
            a, b, c, d = u[e1], v[e1], u[e2], v[e2]
            wts_change = wts[e1] - wts[e2]
            delta_energy = (
                2
                * wts_change
                * (
                    2 * wts_change
                    + (s[a] - sb[a])
                    + (s[b] - sb[b])
                    - (s[c] - sb[c])
                    - (s[d] - sb[d])
                )
            ) / n

            # permutation acceptance criterion
            if delta_energy < 0 or prob < np.e ** (-(delta_energy) / temp):
                sb[[a, b]] -= wts_change
                sb[[c, d]] += wts_change
                wts[[e1, e2]] = wts[[e2, e1]]

                energy = np.mean((sb - s) ** 2)

                if energy < energymin:
                    energymin = energy
                    wtsmin = wts.copy()
                naccept = naccept + 1

        # temperature update
        temp = temp * frac
        if verbose:
            print(
                "\nstage {:d}, temp {:.5f}, best energy {:.5f}, "
                "frac of accepted moves {:.3f}".format(
                    istage, temp, energymin, naccept / niter
                )
            )

    B = np.zeros((n, n))
    B[(u, v)] = wtsmin
    B = B + B.T

    return B, energymin


def strength_preserving_rand_sa_dir(
    A,
    rewiring_iter=10,
    nstage=100,
    niter=10000,
    temp=1000,
    frac=0.5,
    energy_type="sse",
    energy_func=None,
    connected=True,
    verbose=False,
    seed=None,
):
    """
    Strength-preserving network randomization using simulated annealing.

    Randomize a directed weighted network, while preserving
    the in- and out-degree and strength sequences using simulated annealing.

    Parameters
    ----------
    A : (N, N) array-like
        Directed weighted connectivity matrix
    rewiring_iter : int, optional
        Rewiring parameter. Default = 10.
        Each edge is rewired approximately rewiring_iter times.
    nstage : int, optional
        Number of annealing stages. Default = 100.
    niter : int, optional
        Number of iterations per stage. Default = 10000.
    temp : float, optional
        Initial temperature. Default = 1000.
    frac : float, optional
        Fractional decrease in temperature per stage. Default = 0.5.
    energy_type: str, optional
        Energy function to minimize. Can be either:
            'sse': Sum of squared errors between strength sequence vectors
                   of the original network and the randomized network
            'max': Maximum absolute error
            'mae': Mean absolute error
            'mse': Mean squared error
            'rmse': Root mean squared error
        Default = 'sse'.
    energy_func: callable, optional
        Callable with two positional arguments corresponding to
        two strength sequence numpy arrays that returns an energy value.
        Overwrites “energy_type”.
        See “energy_type” for specifying a predefined energy type instead.
    connected: bool, optional
        Whether to ensure connectedness of the randomized network.
        Default = True.
    verbose: bool, optional
        Whether to print status to screen at the end of every stage.
        Default = False.
    seed: float, optional
        Random seed. Default = None.

    Returns
    -------
    B : (N, N) array-like
        Randomized connectivity matrix
    min_energy : float
        Minimum energy obtained by annealing

    Notes
    -----
    Uses Maslov & Sneppen rewiring model to produce a
    surrogate connectivity matrix, B, with the same
    size, density, and in- and out-degree sequences as A.
    The weights are then permuted to optimize the
    match between the strength sequences of A and B
    using simulated annealing.
    Both in- and out-strengths are preserved.

    This function is adapted from a function written in MATLAB
    by Richard Betzel.

    References
    ----------
    Misic, B. et al. (2015) Cooperative and Competitive Spreading Dynamics
    on the Human Connectome. Neuron.
    Rubinov, M. (2016) Constraints and spandrels of interareal connectomes.
    Nature Communications.
    Milisav, F. et al. (2024) A simulated annealing algorithm for
    randomizing weighted networks.
    """
    try:
        A = np.asarray(A)
    except TypeError as err:
        msg = "A must be array_like. Received: {}.".format(type(A))
        raise TypeError(msg) from err

    if frac > 1 or frac <= 0:
        msg = "frac must be between 0 and 1. " "Received: {}.".format(frac)
        raise ValueError(msg)

    rs = check_random_state(seed)

    n = A.shape[0]
    s_in = np.sum(A, axis=0)  # in-strengths of A
    s_out = np.sum(A, axis=1)  # out-strengths of A

    # Maslov & Sneppen rewiring
    if connected:
        B = bct.randmio_dir_connected(A, rewiring_iter, seed=seed)[0]
    else:
        B = bct.randmio_dir(A, rewiring_iter, seed=seed)[0]

    u, v = B.nonzero()  # nonzero indices of B
    wts = B[(u, v)]  # nonzero values of B
    m = len(wts)
    sb_in = np.sum(B, axis=0)  # in-strengths of B
    sb_out = np.sum(B, axis=1)  # out-strengths of B

    if energy_func is not None:
        energy = energy_func(s_in, sb_in) + energy_func(s_out, sb_out)
    elif energy_type == "sse":
        energy = np.sum((s_in - sb_in) ** 2) + np.sum((s_out - sb_out) ** 2)
    elif energy_type == "max":
        energy = np.max(np.abs(s_in - sb_in)) + np.max(np.abs(s_out - sb_out))
    elif energy_type == "mae":
        energy = np.mean(np.abs(s_in - sb_in)) + np.mean(np.abs(s_out - sb_out))
    elif energy_type == "mse":
        energy = np.mean((s_in - sb_in) ** 2) + np.mean((s_out - sb_out) ** 2)
    elif energy_type == "rmse":
        energy = np.sqrt(np.mean((s_in - sb_in) ** 2)) + np.sqrt(
            np.mean((s_out - sb_out) ** 2)
        )
    else:
        msg = (
            "energy_type must be one of 'sse', 'max', "
            "'mae', 'mse', or 'rmse'. Received: {}.".format(energy_type)
        )
        raise ValueError(msg)

    energymin = energy
    wtsmin = wts.copy()

    if verbose:
        print("\ninitial energy {:.5f}".format(energy))

    for istage in tqdm(range(nstage), desc="annealing progress"):
        naccept = 0
        for _ in range(niter):
            # permutation
            e1 = rs.randint(m)
            e2 = rs.randint(m)

            a, b = u[e1], v[e1]
            c, d = u[e2], v[e2]

            sb_prime_in = sb_in.copy()
            sb_prime_out = sb_out.copy()
            sb_prime_in[b] = sb_prime_in[b] - wts[e1] + wts[e2]
            sb_prime_out[a] = sb_prime_out[a] - wts[e1] + wts[e2]
            sb_prime_in[d] = sb_prime_in[d] - wts[e2] + wts[e1]
            sb_prime_out[c] = sb_prime_out[c] - wts[e2] + wts[e1]

            if energy_func is not None:
                energy_prime = energy_func(sb_prime_in, s_in) + energy_func(
                    sb_prime_out, s_out
                )
            elif energy_type == "sse":
                energy_prime = np.sum((sb_prime_in - s_in) ** 2) + np.sum(
                    (sb_prime_out - s_out) ** 2
                )
            elif energy_type == "max":
                energy_prime = np.max(np.abs(sb_prime_in - s_in)) + np.max(
                    np.abs(sb_prime_out - s_out)
                )
            elif energy_type == "mae":
                energy_prime = np.mean(np.abs(sb_prime_in - s_in)) + np.mean(
                    np.abs(sb_prime_out - s_out)
                )
            elif energy_type == "mse":
                energy_prime = np.mean((sb_prime_in - s_in) ** 2) + np.mean(
                    (sb_prime_out - s_out) ** 2
                )
            elif energy_type == "rmse":
                energy_prime = np.sqrt(np.mean((sb_prime_in - s_in) ** 2)) + np.sqrt(
                    np.mean((sb_prime_out - s_out) ** 2)
                )
            else:
                msg = (
                    "energy_type must be one of 'sse', 'max', "
                    "'mae', 'mse', or 'rmse'. "
                    "Received: {}.".format(energy_type)
                )
                raise ValueError(msg)

            # permutation acceptance criterion
            if energy_prime < energy or rs.rand() < np.exp(
                -(energy_prime - energy) / temp
            ):
                sb_in = sb_prime_in.copy()
                sb_out = sb_prime_out.copy()
                wts[[e1, e2]] = wts[[e2, e1]]
                energy = energy_prime
                if energy < energymin:
                    energymin = energy
                    wtsmin = wts.copy()
                naccept = naccept + 1

        # temperature update
        temp = temp * frac
        if verbose:
            print(
                "\nstage {:d}, temp {:.5f}, best energy {:.5f}, "
                "frac of accepted moves {:.3f}".format(
                    istage, temp, energymin, naccept / niter
                )
            )

    B = np.zeros((n, n))
    B[(u, v)] = wtsmin

    return B, energymin
