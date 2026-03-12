# -*- coding: utf-8 -*-
"""
Network assortativity and distance-preserving surrogates
========================================================

This example demonstrates how to calculate network assortativity and generate
distance-preserving surrogate connectomes for null hypothesis testing. We'll
use structural connectivity data and myelin maps from
`Bazinet et al. (2023) <https://doi.org/10.1038/s41467-023-38585-4>`_ to show
how brain network topology relates to spatial patterns of cortical features.
"""

###############################################################################
# First, let's fetch the structural connectivity and myelin data from
# `Bazinet et al. (2023) <https://doi.org/10.1038/s41467-023-38585-4>`_.
# This dataset contains a 400-parcel structural
# connectome and corresponding T1w/T2w myelin maps:

import importlib
import numpy as np
import pickle
from netneurotools.datasets import fetch_bazinet_assortativity

bazinet_path = fetch_bazinet_assortativity()

with open(f'{bazinet_path}/data/human_SC_s400.pickle', 'rb') as file:
    human_data = pickle.load(file)

myelin = human_data['t1t2']
SC = human_data['adj']

print(f'Structural connectivity shape: {SC.shape}')
print(f'Myelin data shape: {myelin.shape}')

###############################################################################
# Let's visualize the structural connectivity matrix. This shows the weighted
# connections between all pairs of brain regions:

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(SC, cmap='Greys', vmin=SC[SC > 0].min(), vmax=SC.max())
ax.set(xlabel='ROI', ylabel='ROI', title='Structural Connectivity')
fig.colorbar(im, ax=ax)

###############################################################################
# To visualize the myelin data on the cortical surface, we need to project
# the parcellated data to the vertex level. We'll use the Schaefer 400-parcel
# atlas to do this mapping:

from netneurotools.datasets import fetch_schaefer2018
from netneurotools.interface import parcels_to_vertices

parc = fetch_schaefer2018('fsaverage')['400Parcels7Networks']
myelin_hemi = (myelin[:200], myelin[200:])
myelin_vertex, _, _ = parcels_to_vertices(myelin_hemi, parc, hemi='both')

###############################################################################
# Now we can plot the myelin data on the inflated cortical surface using
# PyVista:

# This is not run in the example to avoid issues with PyVista dependencies, but
# you can run this code in your local environment if you have PyVista installed.

# from netneurotools.plotting import pv_plot_surface

# pv_plot_surface(
#     myelin_vertex,
#     'fsaverage',
#     'inflated',
#     hemi='both',
#     cmap="Spectral_r",
#     clim=[np.nanmin(myelin_vertex), np.nanmax(myelin_vertex)],
#     cbar_title='T1w/T2w ratio'
# )

###############################################################################
# Network assortativity measures the tendency for nodes with similar features
# to be connected. Let's calculate the assortativity of the myelin distribution
# with respect to the structural connectivity network:

from netneurotools.metrics import assortativity_und

assort = assortativity_und(myelin, SC, use_numba=False)
print(f'Network assortativity: {assort:.3f}')

###############################################################################
# We can visualize this by creating a scatterplot showing the myelin values
# of connected node pairs, with point colors indicating connection strength:

SC_E = SC > 0
Xi = np.repeat(myelin[np.newaxis, :], 400, axis=0)
Xj = np.repeat(myelin[:, np.newaxis], 400, axis=1)
sort_ids = np.argsort(SC[SC_E])

fig, ax = plt.subplots(figsize=(6, 6))
scatter = ax.scatter(
    Xi[SC_E][sort_ids],
    Xj[SC_E][sort_ids],
    c=SC[SC_E][sort_ids],
    cmap='Greys',
    s=3,
    rasterized=True
)
ax.set(xlabel='Myelin (node i)', ylabel='Myelin (node j)')
ax.set_title(f'Assortativity = {assort:.3f}')
fig.colorbar(scatter, ax=ax, label='Connection strength')

###############################################################################
# To test whether this assortativity is significant, we need null models that
# preserve key topological properties. We'll generate distance-preserving
# surrogate networks that maintain the degree distribution and spatial
# embedding of connections.
#
# First, let's calculate the Euclidean distances between parcel centroids:

from scipy.spatial.distance import cdist

get_parcel_centroids = importlib.import_module(
    'neuromaps.nulls.spins'
).get_parcel_centroids
fetch_atlas = importlib.import_module('neuromaps.datasets').fetch_atlas
neuromaps_images = importlib.import_module('neuromaps.images')
relabel_gifti = neuromaps_images.relabel_gifti
annot_to_gifti = neuromaps_images.annot_to_gifti

parc_gii = [relabel_gifti(annot_to_gifti(parc_hemi))[0] for parc_hemi in parc]
fsaverage_atlas = fetch_atlas('fsaverage', '164k')['pial']
parc_centroids, _ = get_parcel_centroids(fsaverage_atlas, parc_gii)
parc_dist = cdist(parc_centroids, parc_centroids)

print(f'Distance matrix shape: {parc_dist.shape}')

###############################################################################
# This is also a good point to make the workflow boundaries explicit. The
# tractography-derived connectome, parcel geometry, and centroid estimation all
# come from upstream resources, while ``netneurotools`` provides the analysis
# interface points used here: annotation-based assortativity, parcel/vertex
# conversion, and distance-preserving null models.

fig, ax = plt.subplots(figsize=(11, 2.8))
ax.axis('off')

boxes = [
    (0.02, 0.55, 0.20, 0.26, "Upstream\ntractography +\nparcellation", "#e9ecef"),
    (0.27, 0.55, 0.18, 0.26, "Surface geometry\n+ centroids\n(neuromaps)", "#e9ecef"),
    (0.50, 0.55, 0.18, 0.26, "netneurotools\ndatasets / interface", "#d8f3dc"),
    (
        0.73,
        0.55,
        0.22,
        0.26,
        "netneurotools\nassortativity +\nnull rewiring",
        "#d8f3dc",
    ),
    (0.73, 0.14, 0.22, 0.20, "Empirical vs. null\ninterpretation", "#fff3bf"),
]

for x0, y0, width, height, label, facecolor in boxes:
    rect = plt.Rectangle((x0, y0), width, height, facecolor=facecolor,
                         edgecolor='black', linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x0 + width / 2, y0 + height / 2, label,
            ha='center', va='center', fontsize=10)

arrowprops = dict(arrowstyle='->', lw=1.5, color='black')
ax.annotate('', xy=(0.27, 0.68), xytext=(0.22, 0.68), arrowprops=arrowprops)
ax.annotate('', xy=(0.50, 0.68), xytext=(0.45, 0.68), arrowprops=arrowprops)
ax.annotate('', xy=(0.73, 0.68), xytext=(0.68, 0.68), arrowprops=arrowprops)
ax.annotate('', xy=(0.84, 0.34), xytext=(0.84, 0.55), arrowprops=arrowprops)

ax.set_title('Workflow context for this example', fontsize=11)

###############################################################################
# Now we'll generate distance-preserving surrogate connectomes. This process
# rewires the network while preserving both the degree distribution and the
# relationship between connection probability and distance.
#
# Note: Generating many surrogates can be time-consuming. For this example,
# we'll create just a few surrogates:

from netneurotools.networks import match_length_degree_distribution

n_surrogates = 10
surr_all = np.zeros((n_surrogates, 400, 400))

for i in range(n_surrogates):
    surr_all[i] = match_length_degree_distribution(SC, parc_dist)[1]

print(f'Generated {n_surrogates} surrogate networks')

###############################################################################
# Let's visualize a few of these surrogate connectomes to see how they compare
# to the empirical network:

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes):
    im = ax.imshow(
        surr_all[i],
        cmap='Greys',
        vmin=surr_all[i][surr_all[i] > 0].min(),
        vmax=surr_all[i].max()
    )
    ax.set_title(f'Surrogate {i + 1}')
    ax.set(xlabel='ROI', ylabel='ROI')
fig.colorbar(im, ax=axes.ravel().tolist(), label='Connection strength')
plt.tight_layout()

###############################################################################
# Now we'll calculate the assortativity for each surrogate network to create
# a null distribution:

assort_surr = np.zeros(n_surrogates)
for i in range(n_surrogates):
    assort_surr[i] = assortativity_und(myelin, surr_all[i])

print(f'Mean surrogate assortativity: {np.mean(assort_surr):.3f}')
print(f'Std surrogate assortativity: {np.std(assort_surr):.3f}')

###############################################################################
# Finally, we can compare the empirical assortativity to the null distribution
# using a boxplot. This visualization shows whether the observed assortativity
# is significantly different from what we'd expect by chance:

fig, ax = plt.subplots(figsize=(4, 6))

flierprops = dict(
    marker='+',
    markerfacecolor='lightgray',
    markeredgecolor='lightgray'
)

bplot = ax.boxplot(
    assort_surr,
    widths=0.3,
    patch_artist=True,
    showfliers=True,
    showcaps=False,
    flierprops=flierprops
)

for box in bplot['boxes']:
    box.set(facecolor='lightgray', edgecolor='black')
for median in bplot['medians']:
    median.set(color='black', linewidth=2)

ax.scatter(1, assort, color='red', s=100, zorder=3, label='Empirical')
ax.set_xticks([1])
ax.set_xticklabels(['Surrogates'])
ax.set_ylabel('Assortativity')
ax.set_title('Empirical vs. Null Distribution')
ax.legend()
ax.set_xlim(0.7, 1.3)

plt.tight_layout()

###############################################################################
# The red dot shows the empirical assortativity, while the boxplot shows the
# distribution of assortativity values from the distance-preserving surrogate
# networks. If the empirical value falls outside the null distribution, this
# suggests that the observed pattern of myelin assortativity is not explained
# by the spatial constraints and topology of the connectome alone.
#
# The limitation is equally important: geometry-aware nulls still depend on the
# quality of the upstream surface / centroid information, and generating large
# surrogate ensembles can be computationally expensive.
