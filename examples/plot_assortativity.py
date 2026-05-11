# -*- coding: utf-8 -*-
"""
Network assortativity and distance-preserving surrogates
========================================================

This example demonstrates how to calculate network assortativity and generate
distance-preserving surrogate connectomes for null hypothesis testing. We'll
use structural connectivity data and myelin maps from
`Bazinet et al. (2023) <https://doi.org/10.1038/s41467-023-38585-4>`_ to
explore how brain network topology relates to spatial patterns of cortical
features:

.. figure:: /_static/examples/plot_assortativity_workflow.png
   :alt: Workflow
   :width: 100%
   :align: center
|
"""

###############################################################################
# First, let's fetch the structural connectivity and myelin data from
# `Bazinet et al. (2023) <https://doi.org/10.1038/s41467-023-38585-4>`_.
# This dataset contains a 400-parcel structural connectome with a corresponding
# distance matrix specifying the distance between parcel centroids. It also
# contains a T1w/T2w myelin map:

import numpy as np
import pickle
from netneurotools.datasets import fetch_bazinet_assortativity

bazinet_path = fetch_bazinet_assortativity()

with open(f'{bazinet_path}/data/human_SC_s400.pickle', 'rb') as file:
    human_data = pickle.load(file)

SC = human_data['adj']
dist = human_data['dist']
myelin = human_data['t1t2']

print(f'Structural connectivity shape: {SC.shape}')
print(f'Distance matrix shape: {dist.shape}')
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
# Let's visualize the parcellated myelin data on the cortical surface. The
# structural connectivity and myelin data have been parcellated with the
# Schaefer (400 nodes) parcellation.

from netneurotools.plotting import pv_plot_parcellated_data

# This is necessary for headless rendering when building the sphinx gallery.
# If you are running this code locally, you might not need this line.
import os
os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"

pv_plot_parcellated_data(myelin,
                         'schaefer400x7',
                         cmap='Spectral_r',
                         clim=[np.min(myelin), np.max(myelin)],
                         cbar_title='T1w/T2w ratio',
                         lighting_style='plastic',
                         jupyter_backend="static")

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
# embedding of connections. For this, we need the empirical structural
# connectome and a matrix specifying the distance between each parcel.
#
# Note: Generating many surrogates can be time-consuming. For this example,
# we'll create just a few surrogates:

from netneurotools.networks import match_length_degree_distribution

n_surrogates = 10
surr_all = np.zeros((n_surrogates, 400, 400))

for i in range(n_surrogates):
    surr_all[i] = match_length_degree_distribution(SC, dist)[1]

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
