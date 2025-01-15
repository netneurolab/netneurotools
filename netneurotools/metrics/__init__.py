"""Magics on networks."""


from .bct import (
    # routing
    degrees_und, degrees_dir,
    distance_wei_floyd, retrieve_shortest_path,
    navigation_wu, get_navigation_path_length,
    # diffusion
    communicability_bin, communicability_wei,
    path_transitivity, search_information,
    mean_first_passage_time, diffusion_efficiency,
    resource_efficiency_bin, flow_graph,
    # other
    assortativity,
    matching_ind_und,
    rich_feeder_peripheral
)


from .metrics_utils import (
    _fast_binarize,
    _graph_laplacian,
)


from .spreading import (
    simulate_atrophy
)


from .statistical import (
    network_pearsonr,
    network_pearsonr_pairwise,
    effective_resistance,
    network_polarisation,
    network_variance,
    network_covariance,
)


__all__ = [
    # bct
    'degrees_und', 'degrees_dir',
    'distance_wei_floyd', 'retrieve_shortest_path',
    'navigation_wu', 'get_navigation_path_length',
    'communicability_bin', 'communicability_wei',
    'path_transitivity', 'search_information',
    'mean_first_passage_time', 'diffusion_efficiency',
    'resource_efficiency_bin', 'flow_graph',
    'assortativity', 'matching_ind_und',
    'rich_feeder_peripheral',
    # metrics_utils
    '_fast_binarize', '_graph_laplacian',
    # spreading
    'simulate_atrophy',
    # statistical
    'network_pearsonr', 'network_pearsonr_pairwise',
    'effective_resistance', 'network_polarisation',
    'network_variance', 'network_covariance',
]
