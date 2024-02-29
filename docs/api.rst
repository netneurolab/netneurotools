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
   match_length_degree_distribution
   randmio_und

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

   sort_communities
   plot_mod_heatmap
   plot_conte69
   plot_fslr
   plot_fsaverage
   plot_fsvertex
   plot_point_brain

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
   get_dominance_stats
   network_pearsonr
   network_pearsonr_numba
   network_pearsonr_pairwise
   effective_resistance
   network_polarisation
   network_variance
   network_variance_numba
   network_covariance
   network_covariance_numba

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

   _binarize
   degrees_und
   degrees_dir
   distance_wei_floyd
   retrieve_shortest_path
   communicability_bin
   communicability_wei
   rich_feeder_peripheral
   navigation_wu
   get_navigation_path_length
   search_information
   path_transitivity
   flow_graph
   mean_first_passage_time
   diffusion_efficiency
   resource_efficiency_bin
   matching_ind_und
   _graph_laplacian

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
    fetch_civet
    fetch_conte69
    fetch_fsaverage
    fetch_pauli2018
    fetch_schaefer2018
    fetch_hcp_standards
    fetch_voneconomo

Functions to download real-world datasets

.. autosummary::
   :template: function.rst
   :toctree: generated/

   fetch_connectome
   fetch_mirchi2018
   fetch_vazquez_rodriguez2019

Functions to generate (pseudo-random) datasets

.. autosummary::
   :template: function.rst
   :toctree: generated/

    make_correlated_xy

.. _ref_freesurfer:

:mod:`netneurotools.freesurfer` - FreeSurfer compatibility functions
--------------------------------------------------------------------

.. automodule:: netneurotools.freesurfer
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.freesurfer

.. autosummary::
   :template: function.rst
   :toctree: generated/

   apply_prob_atlas
   find_parcel_centroids
   parcels_to_vertices
   vertices_to_parcels
   spin_data
   spin_parcels

.. _ref_civet:

:mod:`netneurotools.civet` - CIVET compatibility functions
----------------------------------------------------------

.. automodule:: netneurotools.civet
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.civet

.. autosummary::
   :template: function.rst
   :toctree: generated/

   read_civet
   civet_to_freesurfer

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

.. _ref_colors:

:mod:`netneurotools.colors` - Useful colormaps
--------------------------------------------------------------

.. automodule:: netneurotools.colors
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.colors

.. autosummary::
   :template: function.rst
   :toctree: generated/

   available_cmaps