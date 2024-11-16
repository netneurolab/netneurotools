.. _ref_api:

.. currentmodule:: netneurotools

Python Reference API
====================

.. contents:: **List of modules**
   :local:

.. _ref_datasets:

:mod:`netneurotools.datasets` - Automatic dataset fetching
----------------------------------------------------------

.. automodule:: netneurotools.datasets
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.datasets

To download templates

.. autosummary::
   :template: function.rst
   :toctree: generated/


   fetch_fsaverage
   fetch_fsaverage_curated
   fetch_hcp_standards
   fetch_fslr_curated
   fetch_civet
   fetch_civet_curated
   fetch_conte69
   fetch_yerkes19

To download atlases

.. autosummary::
   :template: function.rst
   :toctree: generated/

    fetch_cammoun2012
    fetch_schaefer2018
    fetch_mmpall
    fetch_pauli2018
    fetch_ye2020
    fetch_voneconomo

To download project-related data

.. autosummary::
   :template: function.rst
   :toctree: generated/

   fetch_vazquez_rodriguez2019
   fetch_mirchi2018
   fetch_hansen_manynetworks
   fetch_hansen_receptors
   fetch_hansen_genecognition
   fetch_hansen_brainstemfc
   fetch_shafiei_megfmrimapping
   fetch_shafiei_megdynamics
   fetch_suarez_mami
   fetch_famous_gmat
   fetch_neurosynth


.. _ref_network:

:mod:`netneurotools.networks` - Constructing networks
-----------------------------------------------------

.. automodule:: netneurotools.networks
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.networks

To construct consensus networks

.. autosummary::
   :template: function.rst
   :toctree: generated/

   func_consensus
   struct_consensus

To randomize networks

.. autosummary::
   :template: function.rst
   :toctree: generated/

   randmio_und
   match_length_degree_distribution
   strength_preserving_rand_sa
   strength_preserving_rand_sa_mse_opt
   strength_preserving_rand_sa_dir

Convenient functions

.. autosummary::
   :template: function.rst
   :toctree: generated/

   binarize_network
   threshold_network


.. _ref_plotting:

:mod:`netneurotools.plotting` - Plotting brain data
---------------------------------------------------

.. automodule:: netneurotools.plotting
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.plotting

PySurfer

.. autosummary::
   :template: function.rst
   :toctree: generated/

   plot_conte69
   plot_fslr
   plot_fsaverage
   plot_fsvertex

Pyvista

.. autosummary::
   :template: function.rst
   :toctree: generated/

   pv_plot_surface

matplotlib

.. autosummary::
   :template: function.rst
   :toctree: generated/

   plot_point_brain
   plot_mod_heatmap

Fun color & colormap stuff

.. autosummary::
   :template: function.rst
   :toctree: generated/

   available_cmaps


.. _ref_metrics:

:mod:`netneurotools.metrics` - Calculating graph metrics
--------------------------------------------------------

.. automodule:: netneurotools.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.metrics

Brain network metrics

.. autosummary::
   :template: function.rst
   :toctree: generated/

   degrees_und
   degrees_dir
   distance_wei_floyd
   retrieve_shortest_path
   navigation_wu
   get_navigation_path_length
   communicability_bin
   communicability_wei
   path_transitivity
   search_information
   mean_first_passage_time
   diffusion_efficiency
   resource_efficiency_bin
   flow_graph
   assortativity
   matching_ind_und
   rich_feeder_peripheral

Network spreading

.. autosummary::
   :template: function.rst
   :toctree: generated/

   simulate_atrophy

Statistical network metrics

.. autosummary::
   :template: function.rst
   :toctree: generated/

   network_pearsonr
   network_pearsonr_numba
   network_pearsonr_pairwise
   effective_resistance
   network_polarisation
   network_variance
   network_variance_numba
   network_covariance
   network_covariance_numba


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

   match_cluster_labels
   match_assignments
   reorder_assignments
   find_consensus
   consensus_modularity
   zrand
   get_modularity
   get_modularity_z
   get_modularity_sig


.. _ref_stats:

:mod:`netneurotools.stats` - General statistics functions
---------------------------------------------------------

.. automodule:: netneurotools.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.stats

Correlations

.. autosummary::
   :template: function.rst
   :toctree: generated/

   efficient_pearsonr
   weighted_pearsonr
   make_correlated_xy

Permutation tests

.. autosummary::
   :template: function.rst
   :toctree: generated/

   permtest_1samp
   permtest_rel
   permtest_pearsonr

Regressions

.. autosummary::
   :template: function.rst
   :toctree: generated/

   residualize
   get_dominance_stats


.. _ref_spatial:

:mod:`netneurotools.spatial` - Spatial statistics
-------------------------------------------------

.. automodule:: netneurotools.spatial
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.spatial

Calculating spatial statistics

.. autosummary::
   :template: function.rst
   :toctree: generated/

   morans_i
   local_morans_i


.. _ref_interface:

:mod:`netneurotools.interface` - Interface for external tools
-------------------------------------------------------------

.. automodule:: netneurotools.interface
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.interface

.. autosummary::
   :template: function.rst
   :toctree: generated/


.. _ref_experimental:

:mod:`netneurotools.experimental` - Functions in alpha stage
------------------------------------------------------------

.. automodule:: netneurotools.experimental
   :no-members:
   :no-inherited-members:

.. currentmodule:: netneurotools.experimental

.. autosummary::
   :template: function.rst
   :toctree: generated/


