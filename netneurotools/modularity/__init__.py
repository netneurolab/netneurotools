"""Functions for working with network modularity."""


from .modules import (
    match_cluster_labels,
    match_assignments,
    reorder_assignments,
    agreement_matrix,
    find_consensus,
    consensus_modularity,
    _dummyvar,
    zrand,
    _zrand_partitions,
    get_modularity,
    get_modularity_z,
    get_modularity_sig,
)


__all__ = [
    # modules
    'match_cluster_labels', 'match_assignments', 'reorder_assignments',
    'agreement_matrix', 'find_consensus', 'consensus_modularity', '_dummyvar',
    'zrand', '_zrand_partitions', 'get_modularity', 'get_modularity_z',
    'get_modularity_sig',
]
