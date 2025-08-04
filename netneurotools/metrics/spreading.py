"""
Functions to simulate atrophy on brain networks with a S.I.R. spreading model.

Python version of SIRsimulator, by Ying-Qiu Zheng:
https://github.com/yingqiuz/SIR_simulator

The original matlab version was originally used in the following paper:

Zheng, Y. Q., Zhang, Y., Yau, Y., Zeighami, Y., Larcher, K.,
       Misic, B., & Dagher, A. (2019). Local vulnerability and global
       connectivity jointly shape neurodegenerative disease propagation.
       PLoS biology, 17(11), e3000495.

This Python version has been used in:

Shafiei, G., Bazinet, V., Dadar, M., Manera, A. L., Collins, D. L.,
        Dagher, A., ... & Ducharme, S. (2023). Network structure and
        transcriptomic vulnerability shape atrophy in frontotemporal dementia.
        Brain, 146(1), 321-336.
"""

import numpy as np
from scipy.stats import norm, zscore


def simulate_atrophy(SC_den, SC_len, seed, roi_sizes, T_total=1000, dt=0.1,
                     p_stay=0.5, v=1, trans_rate=1, init_number=1, GBA=None,
                     SNCA=None, k1=0.5, k=0, FC=None):
    """
    Simulate atrophy on a specified network.

    This is a python version of SIRsimulator, by Ying-Qiu Zheng:
    https://github.com/yingqiuz/SIR_simulator [1]_. This python version was
    first used in [2]_.

    Parameters
    ----------
    SC_den: (n, n) ndarray
        Structural connectivity matrix (strength)
    SC_len: (n, n) ndarray
        Structural connectivity matrix (len)
    seed: int
        ID of the node to be used as a seed for the atrophy process
    roi_sizes: (n,) ndarray:
        Size of each ROIs in the parcellation
    T_total: int
        Total time steps of the function
    dt: float
        Size of each time step
    p_stay: float
        The probability of staying in the same region per unit time
    v: float
        Speed of the atrophy process
    trans_rate: float
        A scalar value controlling the baseline infectivity
    init_number: int
        Number of injected misfolded protein
    GBA: (n,) ndarray
        GBA gene expression (clearance of misfolded protein)
    SNCA: (n,) ndarray
        SNCA gene expression (synthesis of misfolded protein)
    k1: float
        Ratio between weight of atrophy accrual due to accumulation of
        misfolded agends vs. weight of atrophy accrual due to deafferation.
        Must be between 0 and 1
    FC: (n, n) ndarray
        Functional connectivity
    k: float
        weight of functional connectivity

    Returns
    -------
    simulated_atrophy: (n_regions, T_total) ndarray
        Trajectory matrix of the simulated atrophy in individual brain regions.

    References
    ----------
    .. [1] Zheng, Y. Q., Zhang, Y., Yau, Y., Zeighami, Y., Larcher, K.,
       Misic, B., & Dagher, A. (2019). Local vulnerability and global
       connectivity jointly shape neurodegenerative disease propagation.
       PLoS biology, 17(11), e3000495.

    .. [2] Shafiei, G., Bazinet, V., Dadar, M., Manera, A. L.,
       Collins, D. L., Dagher, A., ... & Ducharme, S. (2023). Network structure
       and transcriptomic vulnerability shape atrophy in frontotemporal
       dementia. Brain, 146(1), 321-336.
    """
    # set-up syn_control
    syn_control = roi_sizes

    # Simulated spread of normal proteins
    Pnor0, Rnor0 = _normal_spread(SC_den,
                                  SC_len,
                                  syn_control,
                                  dt=dt,
                                  p_stay=p_stay,
                                  GBA=GBA,
                                  SNCA=SNCA,
                                  k=k,
                                  FC=FC)

    # Simulated spread of misfolded atrophy
    Rnor_all, Rmis_all = _mis_spread(SC_den,
                                     SC_len,
                                     seed,
                                     syn_control,
                                     roi_sizes,
                                     Rnor0.copy(),
                                     Pnor0.copy(),
                                     v=v,
                                     dt=dt,
                                     p_stay=p_stay,
                                     trans_rate=trans_rate,
                                     init_number=init_number,
                                     T_total=T_total,
                                     GBA=GBA,
                                     SNCA=SNCA,
                                     k=k,
                                     FC=FC)

    # Estimate atrophy
    simulated_atrophy = _atrophy(SC_den,
                                 Rnor_all,
                                 Rmis_all,
                                 dt=dt,
                                 k1=k1,
                                 k=k,
                                 FC=FC)

    return simulated_atrophy


def _normal_spread(SC_den, SC_len, syn_control, v=1, dt=0.1, p_stay=0.5,
                   GBA=None, SNCA=None, FC=None, k=0):
    """
    Simulate the spread of normal proteins in a brain network.

    Part 1 of SIRsimulator. SIRsimulator being the original code written by
    Ying-Qiu Zheng in Matlab (https://github.com/yingqiuz/SIR_simulator) for
    her PLoS Biology paper [1]_. This Python version was first used in [2]_.

    Parameters
    ----------
    SC_den: (n, n) ndarray
        Structural connectivity matrix (strength)
    SC_len: (n, n) ndarray
        Structural connectivity matrix (len)
    syn_control: (n,) ndarray
        Parameters specifying in how many voxels proteins can be synthesized
        for each brain regions (region size, i.e., ROIsize)
    v: float
        Speed of the atrophy process. Default: 1
    dt: float
        Size of each time step. Default: 0.1
    p_stay: float
        The probability of staying in the same region per unit time.
        Default: 0.5
    GBA: (n,) ndarray
        GBA gene expression (clearance of misfolded protein). If None, then
        GBA expression is uniformly distributed across brain regions.
        Default: None
    SNCA: (n,) ndarray
        SNCA gene expression (synthesis of misfolded protein)/ If None, then
        SNCA expression is uniformly distributed across brain regions.
        Default: None
    FC: (n, n) ndarray
        Functional connectivity. Default: None
    k: float
        weight of functional connectivity.  Default: 0

    Returns
    -------
    Rnor: (n,) ndarray
         The population of normal agents in regions before pathogenic
         spreading.
    Pnor: (n,) ndarray
        The population of normal agents in edges before pathogenic spreading.

    References
    ----------
    .. [1] Zheng, Y. Q., Zhang, Y., Yau, Y., Zeighami, Y., Larcher, K.,
       Misic, B., & Dagher, A. (2019). Local vulnerability and global
       connectivity jointly shape neurodegenerative disease propagation.
       PLoS biology, 17(11), e3000495.

    .. [2] Shafiei, G., Bazinet, V., Dadar, M., Manera, A. L.,
       Collins, D. L., Dagher, A., ... & Ducharme, S. (2023). Network structure
       and transcriptomic vulnerability shape atrophy in frontotemporal
       dementia. Brain, 146(1), 321-336.
    """
    # Compute basic information
    N_regions = len(SC_len)

    # make sure the diag are zero
    np.fill_diagonal(SC_den, 0)
    np.fill_diagonal(SC_len, 0)

    # Create a Fake FC matrix if FC is none
    if FC is not None:
        np.fill_diagonal(FC, 0)
    else:
        FC = np.zeros((N_regions, N_regions))

    # set probabilities of moving from region i to edge (i,j))
    weights = SC_den * np.exp(k * FC)
    weight_str = weights.sum(axis=0)
    weights = (1 - p_stay) * weights + p_stay * np.diag(weight_str)
    weights = weights / weight_str[:, np.newaxis]

    # convert gene expression scores to probabilities
    if GBA is None or not np.any(GBA):
        clearance_rate = norm.cdf(np.zeros((N_regions)))
    else:
        clearance_rate = norm.cdf(zscore(GBA))

    if SNCA is None or not np.any(SNCA):
        synthesis_rate = norm.cdf(np.zeros((N_regions)))
    else:
        synthesis_rate = norm.cdf(zscore(SNCA))

    # Rnor, Pnor store results of single simulation at each time
    Rnor = np.zeros((N_regions, 1))  # number of normal agent in regions
    Pnor = np.zeros((N_regions, N_regions))  # number of normal agent in paths

    # normal alpha-syn growth
    # fill the network with normal proteins
    iter_max = 1000000000
    for _ in range(iter_max):
        # moving process

        # regions towards paths
        # movDrt stores the number of proteins towards each region. i.e.
        # element in kth row lth col denotes the number of proteins in region k
        # moving towards l
        movDrt = np.repeat(Rnor, N_regions, axis=1) * weights * dt
        np.fill_diagonal(movDrt, 0)

        # paths towards regions
        # update moving
        with np.errstate(divide='ignore', invalid='ignore'):
            movOut = (Pnor * v) / SC_len
            movOut[SC_len == 0] = 0

        Pnor = Pnor - movOut * dt + movDrt
        np.fill_diagonal(Pnor, 0)

        Rtmp = Rnor
        Rnor = Rnor + movOut.sum(axis=0)[:, np.newaxis] * dt - movDrt.sum(axis=1)[:, np.newaxis]  # noqa

        # growth process
        Rnor_cleared = Rnor * (1 - np.exp(-clearance_rate * dt))[:, np.newaxis]
        Rnor_synthesized = ((synthesis_rate * syn_control) * dt)[:, np.newaxis]
        Rnor = Rnor - Rnor_cleared + Rnor_synthesized

        if np.all(abs(Rnor - Rtmp) < 1e-7 * Rtmp):
            break

    return Pnor, Rnor


def _mis_spread(SC_den, SC_len, seed, syn_control, ROIsize, Rnor, Pnor, v=1,
                dt=0.1, p_stay=0.5, trans_rate=1, init_number=1, T_total=1000,
                GBA=None, SNCA=None, return_agents_in_paths=False,
                FC=None, k=0):
    """
    Simulate the spread of misfolded proteins in a brain network.

    Part 2 of SIRsimulator. SIRsimulator being the original code written by
    Ying-Qiu Zheng in Matlab (https://github.com/yingqiuz/SIR_simulator) for
    her PLoS Biology paper [1]_. This Python version was first used in [2]_.

    Parameters
    ----------
    SC_den: (n, n) ndarray
        Structural connectivity matrix (strength)
    SC_len: (n, n) ndarray
        Structural connectivity matrix (len)
    seed: int
        ID of the node to be used as a seed for the atrophy process
    syn_control: (n,) ndarray
        Parameters specifying in how many voxels proteins can be synthesized
        for each brain regions (region size, i.e., ROIsize)
    ROIsize: (n,) ndarray:
        Size of each ROIs in the parcellation
    Rnor: (n,) ndarray
         The population of normal agents in regions before pathogenic
         spreading.
    Pnor: (n,) ndarray
        The population of normal agents in edges before pathogenic spreading.
    v: float
        Speed of the atrophy process
    dt: float
        Size of each time step
    p_stay: float
        The probability of staying in the same region per unit time
    trans_rate: float
        A scalar value controlling the baseline infectivity
    init_number: int
        Number of injected misfolded protein
    T_total: int
        Total time steps of the function
    GBA: (n,) ndarray
        GBA gene expression (clearance of misfolded protein)
    SNCA: (n,) ndarray
        SNCA gene expression (synthesis of misfolded protein)
    return_agents_in_paths: Boolean
        Whether the function should return the number of normal and misfolded
        proteins in each path (edge) of the network. This could be
        memory-consuming. Default: False
    FC: (n, n) ndarray
        Functional connectivity
    k: float
        weight of functional connectivity

    Returns
    -------
    Rnor_all: (n_regions, T_total) ndarray
        Trajectory matrix of the distribution of normal proteins across brain
        regions for each individual time points.
    Rmis_all: (n_regions, T_total) ndarray
        Trajectory matrix of the distribution of misfolded proteins across
        brain regions for each individual time points.
    Pnor_all: (n_regions, n_regions, T_total) ndarray
        Trajectory matrices of the distribution of normal proteins across
        network paths (edges) for each individual time points.
    Pmis_all: (n_regions, n_regions, T_total) ndarray
        Trajectory matrices of the distribution of misfolded proteins across
        network paths (edges) for each individual time points.

    References
    ----------
    .. [1] Zheng, Y. Q., Zhang, Y., Yau, Y., Zeighami, Y., Larcher, K.,
       Misic, B., & Dagher, A. (2019). Local vulnerability and global
       connectivity jointly shape neurodegenerative disease propagation.
       PLoS biology, 17(11), e3000495.

    .. [2] Shafiei, G., Bazinet, V., Dadar, M., Manera, A. L.,
       Collins, D. L., Dagher, A., ... & Ducharme, S. (2023). Network structure
       and transcriptomic vulnerability shape atrophy in frontotemporal
       dementia. Brain, 146(1), 321-336.
    """
    # Compute basic information
    N_regions = len(SC_len)

    # make sure the diag is zero
    np.fill_diagonal(SC_den, 0)
    np.fill_diagonal(SC_len, 0)

    # Create a Fake FC matrix if FC is none
    if FC is not None:
        np.fill_diagonal(FC, 0)
    else:
        FC = np.zeros((N_regions, N_regions))

    # set probabilities of moving from region i to edge (i,j))
    weights = SC_den * np.exp(k * FC)
    weight_str = weights.sum(axis=0)
    weights = (1 - p_stay) * weights + p_stay * np.diag(weight_str)
    weights = weights / weight_str[:, np.newaxis]

    # convert gene expression scores to probabilities
    if GBA is None or not np.any(GBA):
        clearance_rate = norm.cdf(np.zeros((N_regions)))
    else:
        clearance_rate = norm.cdf(zscore(GBA))

    if SNCA is None or not np.any(SNCA):
        synthesis_rate = norm.cdf(np.zeros((N_regions)))
    else:
        synthesis_rate = norm.cdf(zscore(SNCA))

    # store the number of normal/misfoled alpha-syn at each time step
    Rnor_all = np.zeros((N_regions, T_total))
    Rmis_all = np.zeros((N_regions, T_total))

    if return_agents_in_paths:
        Pnor_all = np.zeros((N_regions, N_regions, T_total))
        Pmis_all = np.zeros((N_regions, N_regions, T_total))

    # Rnor, Rmis, Pnor, Pmis store results of single simulation at each time
    Rmis = np.zeros((N_regions, 1))  # nb of misfolded agent in regions
    Pmis = np.zeros((N_regions, N_regions))  # nb of misfolded agent in paths

    # misfolded protein spreading process

    # inject misfolded alpha-syn
    Rmis[seed] = init_number
    for t in range(T_total):
        # moving process

        # normal proteins : region -->> paths
        movDrt_nor = np.repeat(Rnor, N_regions, axis=1) * weights * dt
        np.fill_diagonal(movDrt_nor, 0)

        # normal proteins : path -->> regions
        with np.errstate(invalid='ignore'):
            movOut_nor = (Pnor * v) / SC_len
            movOut_nor[SC_len == 0] = 0

        # misfolded proteins: region -->> paths
        movDrt_mis = np.repeat(Rmis, N_regions, axis=1) * weights * dt
        np.fill_diagonal(movDrt_mis, 0)

        # misfolded proteins: paths -->> regions
        with np.errstate(invalid='ignore'):
            movOut_mis = (Pmis * v) / SC_len
            movOut_mis[SC_len == 0] = 0

        # update regions and paths
        Pnor = Pnor - movOut_nor * dt + movDrt_nor
        np.fill_diagonal(Pnor, 0)
        Rnor = Rnor + movOut_nor.sum(axis=0)[:, np.newaxis] * dt - movDrt_nor.sum(axis=1)[:, np.newaxis]  # noqa

        Pmis = Pmis - movOut_mis * dt + movDrt_mis
        np.fill_diagonal(Pmis, 0)
        Rmis = Rmis + movOut_mis.sum(axis=0)[:, np.newaxis] * dt - movDrt_mis.sum(axis=1)[:, np.newaxis]  # noqa

        Rnor_cleared = Rnor * (1 - np.exp(-clearance_rate * dt))[:, np.newaxis]
        Rnor_synthesized = ((synthesis_rate * syn_control) * dt)[:, np.newaxis]
        Rmis_cleared = Rmis * (1 - np.exp(-clearance_rate * dt))[:, np.newaxis]

        # the probability of getting misfolded
        gamma0 = trans_rate / ROIsize
        misProb = 1 - np.exp(-Rmis * gamma0[:, np.newaxis] * dt)

        # Number of newly infected
        N_misfolded = Rnor * (np.exp(-clearance_rate)[:, np.newaxis]) * misProb

        # Update
        Rnor = Rnor - Rnor_cleared - N_misfolded + Rnor_synthesized
        Rmis = Rmis - Rmis_cleared + N_misfolded

        Rnor_all[:, t] = np.squeeze(Rnor)
        Rmis_all[:, t] = np.squeeze(Rmis)

    if return_agents_in_paths:
        Pnor_all[:, :, t] = Pnor
        Pmis_all[:, :, t] = Pmis

        return Rnor_all, Rmis_all, Pnor_all, Pmis_all

    else:
        return Rnor_all, Rmis_all


def _atrophy(SC_den, Rnor_all, Rmis_all, dt=0.1, k1=0.5, k=0, FC=None):
    """
    Estimate the atrophy from the normal and misfolded proteins distributions.

    This function is inspired by code originally written in Matlab by Ying-Qiu
    Zheng (https://github.com/yingqiuz/SIR_simulator) for her PLoS Biology
    paper [1]_ and was first used in [2]_.

    Parameters
    ----------
    SC_den: (n_regions, n_regions) ndarray
        Structural connectivity matrix (strength).
    Rnor_all: (n_regions, T_total) ndarray
        Trajectory matrix of the distribution of normal protein across brain
        regions for each individual time points.
    Rmis_all: (n_regions, T_total) ndarray
        Trajectory matrix of the distribution of misfolded protein across brain
        regions for each individual time points.
    dt: float
        Size of each time step
    k1: float
        Ratio between weight of atrophy accrual due to accumulation of
        misfolded agends vs. weight of atrophy accrual due to deafferation.
        Must be between 0 and 1
    k: float
        weight of functional connectivity
    FC: (n, n) ndarray
        Functional connectivity

    Returns
    -------
    simulated_atrophy : (n_regions, T_total) ndarray
        Trajectory matrix of the simulated atrophy in individual brain regions.

    References
    ----------
    .. [1] Zheng, Y. Q., Zhang, Y., Yau, Y., Zeighami, Y., Larcher, K.,
       Misic, B., & Dagher, A. (2019). Local vulnerability and global
       connectivity jointly shape neurodegenerative disease propagation.
       PLoS biology, 17(11), e3000495.

    .. [2] Shafiei, G., Bazinet, V., Dadar, M., Manera, A. L.,
       Collins, D. L., Dagher, A., ... & Ducharme, S. (2023). Network structure
       and transcriptomic vulnerability shape atrophy in frontotemporal
       dementia. Brain, 146(1), 321-336.
    """
    # Compute basic information
    N_regions = len(SC_den)

    # Create empty matrix if FC is none
    if FC is not None:
        np.fill_diagonal(FC, 0)
    else:
        FC = np.zeros((N_regions, N_regions))

    ratio = Rmis_all / (Rnor_all + Rmis_all)
    ratio[ratio == np.inf] = 0  # remove possible inf

    # atrophy growth
    k2 = 1 - k1
    weights = SC_den * np.exp(k * FC)
    weights = weights / weights.sum(axis=0)[:, np.newaxis]

    # neuronal loss caused by lack of input from neighbouring regions
    ratio_cum = np.matmul(weights,
                          (1 - np.exp(-ratio * dt)))

    # one time step back
    ratio_cum = np.c_[np.zeros((N_regions, 1)), ratio_cum[:, :-1]]
    ratio_cum = k2 * ratio_cum + k1 * (1 - np.exp(-ratio * dt))

    simulated_atrophy = np.cumsum(ratio_cum, axis=1)

    return simulated_atrophy
