# -*- coding: utf-8 -*-
"""
Spin-tests for significance testing
===================================

This example shows how to perform "spin-tests" (Alexander-Bloch et al., 2018)
to assess whether two brain patterns are correlated above and beyond what would
be expected from spatially-autocorrelated null models.
"""

###############################################################################
# First let's generate some spatially autocorrelated data. We'll set a random
# seed to make sure that things are reproducible:

import numpy as np
rs = np.random.RandomState(1234)

from netneurotools import datasets
x, y = datasets.make_correlated_xy(size=68, seed=rs)

###############################################################################
# We can correlate the vectors to see how related they are:

from scipy.stats import pearsonr
r, p = pearsonr(x, y)
print(r, p)

###############################################################################
# These vectors are quite correlated, and the correlation appears to be very
# significant. However, there's a possibility that the correlation of these
# two vectors is inflated by the spatial organization of the brain. We want to
# create a null distribution of correlations via permutation to assess whether
# this correlation is truly significant or not.
#
# We could randomly permute one of the vectors and regenerate the correlation:

r_perm = np.zeros((1000,))
for perm in range(1000):
    r_perm[perm] = pearsonr(rs.permutation(x), y)[0]
p_perm = (np.sum(r_perm > r) + 1) / (len(r_perm) + 1)
print(p_perm)

###############################################################################
# The permuted p-value suggests that our data are, indeed, highly correlated.
# Unfortunately this does not take into account that the data are constrained
# by a spatial toplogy (i.e., the brain) and thus are not entirely
# exchangeable as is assumed by a normal permutation test.
#
# Instead, we can resample the data by thinking about the brain as a sphere
# and considering random rotations of this sphere. If we rotate the data and
# resample datapoints based on their rotated values, we can generate a null
# distribution that is more appropriate.
#
# To do this we need the spatial coordinates of our brain regions as well as
# an array indicating to which hemisphere each region belongs. We'll use one
# of the parcellations commonly employed in the lab (Cammoun et al., 2012).
# First, we'll fetch the left and right hemisphere FreeSurfer-style annotation
# files for this parcellation:

lhannot, rhannot = datasets.fetch_cammoun2012('surface')['scale033']

###############################################################################
# Then we'll find the centroids of this parcellation defined on the surface
# of a sphere:

from netneurotools import freesurfer
coords, hemi = freesurfer.find_fsaverage_centroids(lhannot, rhannot)
print(coords.shape, hemi.shape)

###############################################################################
# Next, we generate a resampling array based on this "rotation" null model:

from netneurotools import stats
spin = stats.gen_spinsamples(coords, hemi)
print(spin.shape)

###############################################################################
# We can use this to resample one of our data vectors and regenerate the
# correlations:

r_spinperm = np.zeros((1000,))
for perm in range(1000):
    r_spinperm[perm] = pearsonr(x[spin[:, perm]], y)[0]
p_spinperm = (np.sum(r_perm > r) + 1) / (len(r_perm) + 1)
print(p_spinperm)

###############################################################################
# We see that the original correlation is still significant!
