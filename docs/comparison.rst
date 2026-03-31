.. _comparison:

Comparison with existing toolboxes
==================================

Relative to existing toolboxes, netneurotools occupies a distinct niche: it
provides small, modular, decoupled functions that are agnostic to upstream
preprocessing choices, making it straightforward to integrate into diverse
analysis pipelines. Contribution and updating cycles are also faster than in
more monolithic packages, and bug reporting is handled transparently through a
public issue tracker.

Original data & methods developed in our lab
--------------------------------------------

Data fetching
~~~~~~~~~~~~~

A set of project fetching functions in :mod:`netneurotools.datasets` download
the data and associated code used in a number of publications from our lab.

The package includes carefully curated data fetching functions covering common
templates (:func:`~netneurotools.datasets.fetch_fsaverage`,
:func:`~netneurotools.datasets.fetch_fsaverage_curated`,
:func:`~netneurotools.datasets.fetch_fslr_curated`,
:func:`~netneurotools.datasets.fetch_civet`,
:func:`~netneurotools.datasets.fetch_civet_curated`,
:func:`~netneurotools.datasets.fetch_conte69`), and popular atlases
(:func:`~netneurotools.datasets.fetch_schaefer2018`,
:func:`~netneurotools.datasets.fetch_cammoun2012`,
:func:`~netneurotools.datasets.fetch_mmpall`,
:func:`~netneurotools.datasets.fetch_pauli2018`,
:func:`~netneurotools.datasets.fetch_tian2020msa`,
:func:`~netneurotools.datasets.fetch_voneconomo`).

Network measures
~~~~~~~~~~~~~~~~

Original network statistics implemented in :mod:`netneurotools.metrics`:

* :func:`~netneurotools.metrics.assortativity_und`: Calculate annotation-based
  assortativity for undirected networks.
* :func:`~netneurotools.metrics.assortativity_dir`: Calculate annotation-based
  assortativity for directed networks.
* :func:`~netneurotools.metrics.network_pearsonr`: Calculate network-weighted
  Pearson correlation between two annotation vectors.
* :func:`~netneurotools.metrics.effective_resistance`: Calculate the effective
  resistance matrix for a weighted graph.
* :func:`~netneurotools.metrics.network_polarisation`: Calculate ideological
  polarisation of a distribution on a network.
* :func:`~netneurotools.metrics.network_variance`: Calculate network-weighted
  variance of a distribution over nodes.
* :func:`~netneurotools.metrics.network_covariance`: Calculate network-weighted
  covariance of a joint distribution over nodes.

Network community operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented in :mod:`netneurotools.modularity`:

* :func:`~netneurotools.modularity.match_cluster_labels`: Align cluster labels in
  one solution to best match a target solution.
* :func:`~netneurotools.modularity.match_assignments`: Re-label clusters across
  multiple assignment columns to best match a target.
* :func:`~netneurotools.modularity.reorder_assignments`: Relabel and reorder rows
  and columns of an assignment matrix for visualization.

Agent-based simulation on networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`~netneurotools.metrics.simulate_atrophy`: Simulate atrophy spreading on
  a network.

Network null models and consensus construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented in :mod:`netneurotools.networks`:

* :func:`~netneurotools.networks.match_length_degree_distribution`: Generate
  degree- and edge length-preserving surrogate connectomes.
* :func:`~netneurotools.networks.strength_preserving_rand_sa`: Strength-preserving
  network randomization using simulated annealing.
* :func:`~netneurotools.networks.struct_consensus`: Build a distance-dependent
  group consensus structural connectivity matrix.
* :func:`~netneurotools.networks.func_consensus`: Build a thresholded group
  consensus functional connectivity matrix via bootstrapping.

Brain data visualization
~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented in :mod:`netneurotools.plotting`:

* :func:`~netneurotools.plotting.pv_plot_surface` and ``pv_plot_subcortex``: Plot
  surface and subcortical data using PyVista.
* :func:`~netneurotools.plotting.plot_mod_heatmap`: Plot data as a heatmap with
  borders drawn around communities.
* :func:`~netneurotools.plotting.plot_point_brain`: Plot data as a cloud of
  points in 3D space based on specified coordinates.

Neuroimaging data file format manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented in :mod:`netneurotools.interface`:

Surface–parcel mapping:

* :func:`~netneurotools.interface.vertices_to_parcels`: Map vertex-level data to
  parcel-level summaries.
* :func:`~netneurotools.interface.parcels_to_vertices`: Map parcel-level data back
  to vertex-level representation.

CIFTI:

* :func:`~netneurotools.interface.describe_cifti`: Print a summary of the
  structure of a CIFTI file.
* :func:`~netneurotools.interface.extract_cifti_volume`: Extract volumetric
  (subcortical) data from a CIFTI file.
* :func:`~netneurotools.interface.extract_cifti_surface`: Extract surface
  (cortical) data from a CIFTI file.
* :func:`~netneurotools.interface.extract_cifti_labels`: Extract parcel label
  information from a CIFTI axis.
* :func:`~netneurotools.interface.extract_cifti_surface_labels`: Extract surface
  parcel labels from a CIFTI axis.
* :func:`~netneurotools.interface.deconstruct_cifti`: Decompose a CIFTI file into
  its surface and volume components.

FreeSurfer:

* :func:`~netneurotools.interface.extract_annot_labels`: Extract parcel label
  names from a FreeSurfer annotation file.

GIFTI:

* :func:`~netneurotools.interface.extract_gifti_labels`: Extract parcel label
  names from a GIFTI parcellation file.

Original, highly optimized Python implementations
---------------------------------------------------

Methods originally in MATLAB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following graph-theoretic measures are implemented in
:mod:`netneurotools.metrics`:

* :func:`~netneurotools.metrics.navigation_wu`: Compute network navigation using
  greedy routing toward the target node.
* :func:`~netneurotools.metrics.communicability_bin`: Compute communicability
  between pairs of nodes in a binary adjacency matrix.
* :func:`~netneurotools.metrics.communicability_wei`: Compute communicability
  between pairs of nodes in a weighted adjacency matrix.
* :func:`~netneurotools.metrics.path_transitivity`: Calculate the density of
  local detours (triangles) available along shortest paths between node pairs.
* :func:`~netneurotools.metrics.resource_efficiency_bin`: Calculate resource
  efficiency and shortest-path probability for binary networks.
* :func:`~netneurotools.metrics.flow_graph`: Calculate the flow graph of a
  network at a given Markov time.
* :func:`~netneurotools.metrics.matching_ind_und`: Calculate the undirected
  matching index (similarity between connectivity profiles).
* :func:`~netneurotools.metrics.rich_feeder_peripheral`: Calculate connectivity
  statistics for rich, feeder, and peripheral edges.

Network community consensus implemented in :mod:`netneurotools.modularity`:

* :func:`~netneurotools.modularity.find_consensus`: Find consensus cluster labels
  from multiple partitioning solutions.
* :func:`~netneurotools.modularity.consensus_modularity`: Run the Louvain
  algorithm repeatedly and derive consensus community assignments.
* :func:`~netneurotools.modularity.get_modularity`: Calculate modularity
  contribution for each community.

The following functions were originally distributed in MATLAB and have several
underperforming Python implementations (bctpy, brainconn), implemented in
:mod:`netneurotools.metrics` and :mod:`netneurotools.networks`:

* :func:`~netneurotools.networks.randmio_und`: Generate randomized networks
  preserving the degree distribution.
* :func:`~netneurotools.metrics.degrees_und`: Calculate degree for undirected
  networks.
* :func:`~netneurotools.metrics.degrees_dir`: Calculate in- and out-degree for
  directed networks.
* :func:`~netneurotools.metrics.search_information`: Calculate search information
  — the bits a random walker needs to follow the shortest path between nodes.
* :func:`~netneurotools.metrics.mean_first_passage_time`: Calculate mean first
  passage time between all node pairs for a random walker.
* :func:`~netneurotools.metrics.diffusion_efficiency`: Calculate diffusion
  efficiency (inverse of mean first passage time) across node pairs.

Methods originally in R
~~~~~~~~~~~~~~~~~~~~~~~~

Implemented in :mod:`netneurotools.spatial`:

* :func:`~netneurotools.spatial.morans_i`: Calculate global Moran's I for spatial
  autocorrelation.
* :func:`~netneurotools.spatial.local_morans_i`: Calculate local Moran's I values
  for each node.
* :func:`~netneurotools.spatial.gearys_c`: Calculate global Geary's C for spatial
  autocorrelation.
* :func:`~netneurotools.spatial.local_gearys_c`: Calculate local Geary's C values
  for each node.
* :func:`~netneurotools.spatial.lees_l`: Calculate Lee's L for bivariate spatial
  autocorrelation.
* :func:`~netneurotools.spatial.local_lees_l`: Calculate local Lee's L values for
  each node pair.

Highly optimized Python implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented in :mod:`netneurotools.stats`:

* :func:`~netneurotools.stats.get_dominance_stats`: Compute dominance analysis
  statistics for relative predictor importance in multilinear regression.

These implementations make methods previously inaccessible or poorly ported
accessible to the broader Python ecosystem.

Comparison with specific toolboxes
------------------------------------

GRETNA
~~~~~~

GRETNA is a well-established toolbox for graph-theoretic analysis of brain
networks that additionally covers rsfMRI preprocessing -- users requiring an
end-to-end preprocessing-to-analysis pipeline may find GRETNA a suitable choice
for that workflow. netneurotools, in contrast, focuses on modular,
preprocessing-agnostic analysis functions with richer brain visualization
support, a public contribution pipeline, and continued active development
targeting newer methods. **We are very happy to add your newly developed method to
netneurotools.**

Brain connectivity toolbox (BCT) / bctpy / brainconn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BCT is the most widely used reference implementation for graph-theoretic
measures for brain networks, and bctpy and brainconn provide Python ports of its
functions. netneurotools complements these efforts by offering carefully
optimized Python-native implementations of select measures, along with
additional methods not covered by BCT, making them readily accessible within the
Python scientific ecosystem. **Let us know if there are specific BCT functions you
would like to see implemented or optimized in netneurotools.**

NetworkX
~~~~~~~~

NetworkX is a general-purpose network analysis library not tailored to brain
data. netneurotools complements rather than competes with NetworkX, and is in
fact compatible with it -- the shortest path routine makes use of the
high-quality implementation in NetworkX while preserving an API interface
convenient and familiar for brain network researchers.

Comet Toolbox
~~~~~~~~~~~~~

Comet Toolbox aims at combining functional connectivity estimation and
graph-theoretical analysis into a unified multiverse workflow. netneurotools, in
contrast, provides modular, preprocessing-agnostic functions that are not tied
to a particular analysis paradigm. The two toolboxes are largely complementary:
Comet Toolbox can be used for comprehensive robustness testing across analysis
choices, while netneurotools can be used for more flexible, modular analyses and
visualizations with cutting-edge network metrics and brain surface rendering.
