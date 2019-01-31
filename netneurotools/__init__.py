__all__ = [
    '__author__', '__description__', '__email__', '__license__',
    '__maintainer__', '__packagename__', '__url__', '__version__',
    'consensus_modularity', 'func_consensus', 'struct_consensus'
]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .info import (
    __author__,
    __description__,
    __email__,
    __license__,
    __maintainer__,
    __packagename__,
    __url__,
)

from .modularity import consensus_modularity
from .networks import func_consensus, struct_consensus
