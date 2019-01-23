__all__ = ['__version__', 'metrics', 'stats', 'utils',
           'consensus_modularity', 'func_consensus', 'struct_consensus',
           'plot_mod_heatmap', 'plot_conte69', 'plot_fsaverage']

from .info import (__version__)
from . import metrics, stats, utils
from .modularity import consensus_modularity
from .networks import func_consensus, struct_consensus
from .plotting import plot_mod_heatmap, plot_conte69, plot_fsaverage
