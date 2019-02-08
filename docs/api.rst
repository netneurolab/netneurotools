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
   :toctree: _generated/

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
   :toctree: _generated/

   consensus_modularity
   zrand
   get_modularity
   get_modularity_z
   get_modularity_sig

.. _ref_plotting:

:mod:`netneurotools.plotting` - Plotting brain data
---------------------------------------------------

.. automodule:: netneurotools.plotting
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.plotting

.. autosummary::
   :template: function.rst
   :toctree: _generated/

   plot_mod_heatmap
   plot_conte69
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
   :toctree: _generated/

   gen_spinsamples
   residualize
   get_mad_outliers
   permtest_1samp
   permtest_rel

.. _ref_metrics:

:mod:`netneurotools.metrics` - Calculating graph metrics
--------------------------------------------------------

.. automodule:: netneurotools.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.metrics

.. autosummary::
   :template: function.rst
   :toctree: _generated/

   communicability
   communicability_wei

.. _ref_datasets:

:mod:`netneurotools.datasets` - Automatic dataset fetching
----------------------------------------------------------

.. automodule:: netneurotools.datasets
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.datasets

Functions to download datasets (atlases, templates, etc)

.. autosummary::
   :template: function.rst
   :toctree: _generated/

    fetch_cammoun2012
    fetch_conte69

Functions to generate (pseudo-random) datasets

.. autosummary::
   :template: function.rst
   :toctree: _generated/

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
   :toctree: _generated/

   run
   add_constant
   get_triu
   get_centroids
