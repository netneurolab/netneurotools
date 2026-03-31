# -*- coding: utf-8 -*-
"""
Are brain modules consistent across connectivity modes?
=======================================================

This example reproduces the core workflow used to compare mesoscale community
structure across multiple cortical connectivity modes from `Hansen et al.
(2023) <https://doi.org/10.1371/journal.pbio.3002314>`_. We load seven
connectivity matrices, inspect how the number of
communities changes with resolution, and summarize mode-specific community
assignments.

Compared with the original figure workflow, the final panel here uses
community assignment matrices directly (instead of cortical surface renders) so
that the full example remains lightweight and self-contained.
"""

###############################################################################
# First, fetch the multimodal connectivity dataset and load all Schaefer-400
# connectivity matrices used in the comparison.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

from netneurotools.datasets import fetch_hansen_manynetworks

dataset_root = Path(fetch_hansen_manynetworks())
data_dir = dataset_root / "data" / "Schaefer400"

matrix_files = [
    "gene_coexpression.npy",
    "receptor_similarity.npy",
    "laminar_similarity.npy",
    "metabolic_connectivity.npy",
    "haemodynamic_connectivity.npy",
    "electrophysiological_connectivity.npy",
    "temporal_similarity.npy",
]
mode_labels = [
    "gene",
    "receptor",
    "laminar",
    "metabolic",
    "haemodynamic",
    "electrophysiological",
    "temporal",
]
mode_abbrev = ["gc", "rs", "ls", "mc", "hc", "ec", "ts"]

###############################################################################
# The seven connectivity modes span molecular, microstructural, metabolic,
# haemodynamic, electrophysiological, and temporal domains:
#
# 1. gene coexpression: transcriptional similarity from AHBA
#    (`Hawrylycz et al. (2012) <https://doi.org/10.1038/nature11405>`_).
# 2. receptor similarity: correspondence in neurotransmitter receptor/
#    transporter profiles (`Hansen et al. (2022)
#    <https://doi.org/10.1038/s41593-022-01186-3>`_).
# 3. laminar similarity: similarity of cortical microstructure profiles derived
#    from BigBrain (`Paquola et al. (2019)
#    <https://doi.org/10.1371/journal.pbio.3000284>`_; `Amunts et al. (2013)
#    <https://doi.org/10.1126/science.1235381>`_).
# 4. metabolic connectivity: coupling of dynamic FDG-fPET signals
#    (`Jamadar et al. (2021) <https://doi.org/10.1093/cercor/bhaa393>`_;
#    `Jamadar et al. (2020) <https://doi.org/10.1038/s41597-020-00699-5>`_).
# 5. haemodynamic connectivity: resting-state BOLD fMRI coupling from HCP
#    (`Van Essen et al. (2013)
#    <https://doi.org/10.1016/j.neuroimage.2013.05.041>`_).
# 6. electrophysiological connectivity: MEG-derived connectivity summary across
#    canonical frequency bands (`Van Essen et al. (2013)
#    <https://doi.org/10.1016/j.neuroimage.2013.05.041>`_; `Shafiei et al.
#    (2022) <https://doi.org/10.1371/journal.pbio.3001735>`_).
# 7. temporal profile similarity: similarity of rich BOLD time-series features
#    (`Shafiei et al. (2020) <https://doi.org/10.7554/eLife.62116>`_;
#    `Fulcher et al. (2013) <https://doi.org/10.1098/rsif.2013.0048>`_;
#    `Fulcher and Jones (2017) <https://doi.org/10.1016/j.cels.2017.10.001>`_).
#
# In the source dataset, all networks are represented at Schaefer-400 and
# harmonized for cross-modal comparison (`Schaefer et al. (2018)
# <https://doi.org/10.1093/cercor/bhx179>`_).

A_all = []
for file_name in matrix_files:
    A = np.load(data_dir / file_name)
    np.fill_diagonal(A, np.nan)
    A_all.append(A)

print(f"Loaded {len(A_all)} connectivity modes from: {data_dir}")

###############################################################################
# The matrices in this example are the output of several upstream processing
# streams, each harmonized to the same Schaefer-400 space before entering
# ``netneurotools``. That division of labor is useful to show explicitly:
# ``netneurotools`` is not re-running gene-expression preprocessing, PET map
# alignment, MEG connectivity estimation, or fMRI preprocessing here; it is the
# analysis layer that fetches, organizes, and visualizes the resulting matrices.

fig, ax = plt.subplots(figsize=(11.5, 3.0))
ax.axis("off")

boxes = [
    (
        0.02,
        0.56,
        0.22,
        0.24,
        "AHBA / PET / BigBrain /\nfMRI / MEG / time-series\npreprocessing",
        "#e9ecef",
    ),
    (0.29, 0.56, 0.18, 0.24, "Shared Schaefer-400\nparcel space", "#e9ecef"),
    (0.52, 0.56, 0.18, 0.24, "fetch_hansen_\nmanynetworks", "#d8f3dc"),
    (0.75, 0.56, 0.20, 0.24, "Community analysis\n+ heatmaps", "#d8f3dc"),
    (0.75, 0.16, 0.20, 0.18, "Cross-modal\ncomparison", "#fff3bf"),
]

for x0, y0, width, height, label, facecolor in boxes:
    rect = plt.Rectangle((x0, y0), width, height, facecolor=facecolor,
                         edgecolor="black", linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x0 + width / 2, y0 + height / 2, label,
            ha="center", va="center", fontsize=10)

arrowprops = dict(arrowstyle="->", lw=1.5, color="black")
ax.annotate("", xy=(0.29, 0.68), xytext=(0.24, 0.68), arrowprops=arrowprops)
ax.annotate("", xy=(0.52, 0.68), xytext=(0.47, 0.68), arrowprops=arrowprops)
ax.annotate("", xy=(0.75, 0.68), xytext=(0.70, 0.68), arrowprops=arrowprops)
ax.annotate("", xy=(0.85, 0.34), xytext=(0.85, 0.56), arrowprops=arrowprops)

ax.set_title("Workflow context for multimodal comparison", fontsize=11)

###############################################################################
# Plot one matrix per mode using mode-specific light-to-dark colormaps.

base_colors = [
    "#2ca8a8",  # gene
    "#f2c14e",  # receptor
    "#f08a5d",  # laminar
    "#c792ea",  # metabolic
    "#73bf69",  # haemodynamic
    "#7aa2f7",  # electrophysiological
    "#ff7aa2",  # temporal
]


def make_mode_cmap(hex_color):
    """Create a white-to-color gradient colormap for one connectivity mode."""
    rgb = to_rgb(hex_color)
    vals = np.ones((256, 4))
    vals[:, 0] = np.linspace(1.0, rgb[0], 256)
    vals[:, 1] = np.linspace(1.0, rgb[1], 256)
    vals[:, 2] = np.linspace(1.0, rgb[2], 256)
    return ListedColormap(vals)


fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
axes = axes.ravel()

for i, (A, label) in enumerate(zip(A_all, mode_labels)):
    ax = axes[i]
    vmax = np.nanmean(A) + 3 * np.nanstd(A)
    vmin = np.nanmean(A) - 3 * np.nanstd(A)
    ax.imshow(A, cmap=make_mode_cmap(base_colors[i]), vmin=vmin, vmax=vmax)
    ax.set_title(label, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

# Hide the final empty panel in the 2x4 layout.
axes[-1].axis("off")

###############################################################################
# Next, we'll run community detection across a range of resolution parameters.
# The workflow below applies consensus modularity clustering with negative
# asymmetry null model across 60 resolution parameters (gamma from 0.1 to 6.0).
# We leave it in comments to avoid running a long analysis in the example, but
# you can run it locally if you have the resources. The results are saved to
# disk and loaded in the next section for visualization.

# from netneurotools.modularity import consensus_modularity
# from joblib import Parallel, delayed

# def community_detection(A, gamma_range):
#     """Run consensus modularity clustering for a range of gamma values."""
#     nnodes = len(A)
#     ngamma = len(gamma_range)
#     consensus = np.zeros((nnodes, ngamma))
#     qall = []
#     zrand = []
#     for i, g in enumerate(gamma_range):
#         consensus[:, i], q, z = consensus_modularity(
#             A, g, B='negative_asym'
#         )
#         qall.append(q)
#         zrand.append(z)
#     return consensus, qall, zrand

# # Load and preprocess connectivity matrices
# networks = {
#     "gc": np.load(data_dir / "gene_coexpression.npy"),
#     "rs": np.load(data_dir / "receptor_similarity.npy"),
#     "ls": np.load(data_dir / "laminar_similarity.npy"),
#     "mc": np.load(data_dir / "metabolic_connectivity.npy"),
#     "hc": np.load(data_dir / "haemodynamic_connectivity.npy"),
#     "ec": np.load(data_dir / "electrophysiological_connectivity.npy"),
#     "ts": np.load(data_dir / "temporal_similarity.npy"),
# }
# # Fisher's z-transform and zero diagonal
# for network in networks.keys():
#     networks[network] = np.arctanh(networks[network])
#     networks[network][np.eye(len(networks[network])).astype(bool)] = 0

# # Run community detection in parallel
# gamma_range = [x / 10.0 for x in range(1, 61, 1)]
# output = Parallel(n_jobs=36)(
#     delayed(community_detection)(networks[network], gamma_range)
#     for network in networks.keys()
# )

# # Save results
# results_dir.mkdir(parents=True, exist_ok=True)
# for network_idx, network_name in enumerate(networks.keys()):
#     np.save(
#         results_dir / f"community_assignments_{network_name}.npy",
#         output[network_idx][0]
#     )
#     np.save(
#         results_dir / f"community_qall_{network_name}.npy",
#         np.array(output[network_idx][1])
#     )
#     np.save(
#         results_dir / f"community_zrand_{network_name}.npy",
#         np.array(output[network_idx][2])
#     )

###############################################################################
# To save time, we load precomputed community assignments across a range of
# resolution parameters and inspect the number of detected communities per mode.

results_dir = (
    dataset_root
    / "results"
    / "community_detection_Schaefer400"
)
y_grid = np.linspace(0.1, 6.0, 60)

ci_all = []
for abb in mode_abbrev:
    ci_path = results_dir / f"community_assignments_{abb}.npy"
    ci_all.append(np.load(ci_path))

fig, ax = plt.subplots(figsize=(6.5, 3.2))
for i, (ci, label) in enumerate(zip(ci_all, mode_labels)):
    n_communities = np.max(ci, axis=0)
    ax.plot(y_grid, n_communities, color=base_colors[i], linewidth=2, label=label)

ax.set(
    xlabel="resolution (gamma)",
    ylabel="n. communities",
    ylim=(0, 40),
    title="Community count across resolutions",
)
ax.legend(ncol=2, frameon=False, fontsize=8)

###############################################################################
# Finally, summarize one representative partition per mode (chosen from the
# same resolution indices as the original analysis) using community assignment
# matrices instead of brain-surface renderings.

y_best = [19, 19, 24, 4, 11, 9, 6]

fig, axes = plt.subplots(2, 4, figsize=(12, 4.5), constrained_layout=True)
axes = axes.ravel()

for i, (ci, label) in enumerate(zip(ci_all, mode_labels)):
    ax = axes[i]
    ci_best = ci[:, y_best[i]]
    # Plot as a 1 x N matrix for quick visual comparison across modes.
    ax.imshow(ci_best[np.newaxis, :], aspect="auto", cmap="Spectral")
    ax.set_title(label, fontsize=10, color=base_colors[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"k={int(np.max(ci_best))}", fontsize=8)

axes[-1].axis("off")

###############################################################################
# You can project parcel-wise community assignments to fsaverage vertices and
# render them on an inflated surface.

from netneurotools.datasets import fetch_schaefer2018
from netneurotools.interface import parcels_to_vertices
from netneurotools.plotting import pv_plot_surface

# This is necessary for headless rendering when building the sphinx gallery.
# If you are running this code locally, you might not need this line.
import os
os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"

parc = fetch_schaefer2018("fsaverage")["400Parcels7Networks"]

for i, (ci, label) in enumerate(zip(ci_all, mode_labels)):
    ci_best = ci[:, y_best[i]]
    ci_hemi = (ci_best[:200], ci_best[200:])
    ci_vertex, _, _ = parcels_to_vertices(ci_hemi, parc, hemi="both")

    pv_plot_surface(
        ci_vertex,
        "fsaverage",
        "inflated",
        hemi="both",
        cmap="Spectral",
        cbar_title=f"{label} communities",
        layout="row",
        show_plot=True,
        lighting_style='plastic',
        jupyter_backend="static"
    )

###############################################################################
# Similar to the consensus-clustering example, we can also visualize each mode
# as a community-ordered heatmap using
# :func:`~netneurotools.plotting.plot_mod_heatmap`.

from netneurotools import plotting

fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
axes = axes.ravel()

for i, (A, ci, label) in enumerate(zip(A_all, ci_all, mode_labels)):
    ax = axes[i]
    ci_best = ci[:, y_best[i]]
    plotting.plot_mod_heatmap(
        A,
        ci_best,
        ax=ax,
        cmap="viridis",
        cbar=False,
    )
    ax.set_title(
        f"{label} (k={int(np.max(ci_best))})",
        fontsize=10,
        color=base_colors[i],
    )

axes[-1].axis("off")

###############################################################################
# These panels provide a compact, modality-by-modality view of community
# organization: the curve plot shows how module count evolves with gamma,
# while the assignment matrices show differences in parcel-wise partitioning
# at representative resolutions.
#
# The main limitation is deliberate: this example starts from harmonized,
# publication-ready matrices, so modality-specific preprocessing choices remain
# upstream of the workflow shown here.
