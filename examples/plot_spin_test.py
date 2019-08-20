# -*- coding: utf-8 -*-
"""
Spin-tests for significance testing
===================================

This example shows how to perform "spin-tests" (a la Alexander-Bloch et al.,
2018, NeuroImage) to assess whether two brain patterns are correlated above and
beyond what would be expected from a spatially-autocorrelated null model.

For the original MATLAB toolbox published alongside the paper by
Alexander-Bloch and colleagues refer to https://github.com/spin-test/spin-test.
"""

###############################################################################
# First let's generate some randomly correlated data. We'll set a random seed
# to make sure that things are reproducible:

from netneurotools import datasets
x, y = datasets.make_correlated_xy(size=68, seed=1234)

###############################################################################
# We can correlate the resulting vectors to see how related they are:

from scipy.stats import pearsonr
r, p = pearsonr(x, y)
print(r, p)

###############################################################################
# The p-value suggests that our data are, indeed, highly correlated.
# Unfortunately, when doing this sort of correlation with brain data the null
# model used in generating the p-value does not take into account that the data
# are constrained by a spatial toplogy (i.e., the brain) and thus spatially
# auto-correlated. The p-values will be "inflated" because our true degrees of
# freedom are less than the number of samples we have!
#
# To address this we can use a spatial permutation test (called a "spin test"),
# first introduced by Alexander-Bloch et al., 2018. This test works by thinking
# about the brain as a sphere and considering random rotations of this sphere.
# If we rotate our data and resample datapoints based on their rotated values,
# we can generate a null distribution that is more appropriate to our spatially
# auto-correlated data.
#
# To do this we need the spatial coordinates of our brain regions, as well as
# an array indicating to which hemisphere each region belongs. In this example
# we'll use one of the parcellations commonly employed in the lab (Cammoun et
# al., 2012). First, we'll fetch the left and right hemisphere FreeSurfer-style
# annotation files for this parcellation (using the lowest resolution of the
# parcellation):

lhannot, rhannot = datasets.fetch_cammoun2012('surface')['scale033']

###############################################################################
# Then we'll find the centroids of this parcellation defined on the spherical
# projection of the fsaverage surface. This function will return the xyz
# coordinates (`coords`) for each parcel defined in `lhannot` and `rhannot`, as
# well as a vector identifying to which hemisphere each parcel belongs
# (`hemi`):

from netneurotools import freesurfer
coords, hemi = freesurfer.find_fsaverage_centroids(lhannot, rhannot)
print(coords.shape, hemi.shape)

###############################################################################
# We'll use these coordinates to generate a resampling array based on this idea
# of a "rotation"-based null model. As an example we'll only generate 1000
# rotations (i.e., permutations), but you can easily generate more by changing
# the `n_rotate` parameter. Since these rotations are random we can set a
# `seed` to ensure reproducibility:

from netneurotools import stats
spin, cost = stats.gen_spinsamples(coords, hemi, n_rotate=1000, seed=1234)
print(spin.shape)

###############################################################################
# The function returns both a resampling array (`spins`) as well as a cost
# vector (`cost`) detailing the average distance between each rotated parcel
# and the one to which it was matched.
#
# We'll use `spins` to resample one of our data vectors and regenerate the
# correlations:

import numpy as np
r_spinperm = np.zeros((spin.shape[-1],))
for perm in range(spin.shape[-1]):
    r_spinperm[perm] = pearsonr(x[spin[:, perm]], y)[0]

###############################################################################
# Finally, we'll generate a non-parametric p-value from these correlations:

p_spinperm = (np.sum(r_spinperm > r) + 1) / (len(r_spinperm) + 1)
print(p_spinperm)
