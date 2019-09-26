.. _ref_api:

.. currentmodule:: netneurotools

Python Reference API
====================

.. contents:: **List of modules**
   :local:

.. _ref_network:

:mod:`netneurotools.networks` - Constructing networks
-----------------------------------------------------

.. automodule:: netneurotools.networks
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.networks

.. autosummary::
   :template: function.rst
   :toctree: generated/

   func_consensus
   struct_consensus
   threshold_network
   binarize_network

.. _ref_modularity:

:mod:`netneurotools.modularity` - Calculating network modularity
----------------------------------------------------------------

.. automodule:: netneurotools.modularity
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.modularity

.. autosummary::
   :template: function.rst
   :toctree: generated/

   consensus_modularity
   zrand
   get_modularity
   get_modularity_z
   get_modularity_sig

.. _ref_cluster:

:mod:`netneurotools.cluster` - Working with clusters
----------------------------------------------------

.. automodule:: netneurotools.cluster
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.cluster

.. autosummary::
   :template: function.rst
   :toctree: generated/

   find_consensus
   match_assignments
   reorder_assignments
   match_cluster_labels

.. _ref_plotting:

:mod:`netneurotools.plotting` - Plotting brain data
---------------------------------------------------

.. automodule:: netneurotools.plotting
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.plotting

.. autosummary::
   :template: function.rst
   :toctree: generated/

   plot_mod_heatmap
   plot_conte69
   plot_point_brain
   plot_fsaverage

.. _ref_stats:

:mod:`netneurotools.stats` - General statistics functions
---------------------------------------------------------

.. automodule:: netneurotools.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.stats

.. autosummary::
   :template: function.rst
   :toctree: generated/

   gen_spinsamples
   residualize
   get_mad_outliers
   efficient_pearsonr
   permtest_1samp
   permtest_rel
   permtest_pearsonr

.. _ref_metrics:

:mod:`netneurotools.metrics` - Calculating graph metrics
--------------------------------------------------------

.. automodule:: netneurotools.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.metrics

.. autosummary::
   :template: function.rst
   :toctree: generated/

   communicability_bin
   communicability_wei

.. _ref_datasets:

:mod:`netneurotools.datasets` - Automatic dataset fetching
----------------------------------------------------------

.. automodule:: netneurotools.datasets
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.datasets

Functions to download atlases and templates

.. autosummary::
   :template: function.rst
   :toctree: generated/

    fetch_cammoun2012
    fetch_conte69
    fetch_fsaverage
    fetch_pauli2018

Functions to download real-world datasets

.. autosummary::
   :template: function.rst
   :toctree: generated/

   fetch_mirchi2018

Functions to generate (pseudo-random) datasets

.. autosummary::
   :template: function.rst
   :toctree: generated/

    make_correlated_xy

.. _ref_utils:

:mod:`netneurotools.utils` - Miscellaneous, grab bag utilities
--------------------------------------------------------------

.. automodule:: netneurotools.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.utils

.. autosummary::
   :template: function.rst
   :toctree: generated/

   run
   add_constant
   get_triu
   get_centroids
