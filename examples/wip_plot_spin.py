# -*- coding: utf-8 -*-
"""
Spatial permutations for significance testing
=============================================

This example shows how to perform spatial permutations tests (a.k.a spin-tests;
`Alexander-Bloch et al., 2018, NeuroImage <https://www.ncbi.nlm.nih.gov/pmc/
articles/PMC6095687/>`_) to assess whether two brain patterns are correlated
above and beyond what would be expected from a spatially-autocorrelated null
model.

While the original spin-tests were designed for comparing surface maps, we
generally work with parcellated data in our lab. Using parcellations presents
some novel difficulties in effectively implementing the spin-test, so this
example demonstrates three spin-test methods.

For the original MATLAB toolbox published alongside the paper by
Alexander-Bloch and colleagues refer to https://github.com/spin-test/spin-test.
"""

###############################################################################
# An example dataset
# ------------------
#
# The spin-test assumes that we have two sets of correlated data and that we
# are interested in assessing the degree to which this correlation exceeds a
# spatially-autocorrelated null model. We generate this null by permuting the
# original data by assuming it can be represented on the surface of the sphere
# and "spinning" the sphere.
#
# First, let's get some parcellated spatial maps that we can compare:

from netneurotools import datasets as nndata

data = nndata.fetch_vazquez_rodriguez2019(verbose=0)
rsq, grad = data['rsquared'], data['gradient']

###############################################################################
# The above function returns the :math:`R^{2}` values of a structure-function
# linear regression model for each parcel (``rsq``) as well as the scores of
# each parcel along the first gradient computed from diffusion map embedding
# of a functional connectivity matrix (``grad``). (Refer to `Vazquez-Rodriguez
# et al., 2019 <https://www.pnas.org/content/early/2019/09/27/1903403116>`_ for
# more information on these variables.)
#
# These two vectors contain values for 1000 brain regions (a high-resolution
# sub-division of the Desikian-Killiany atlas; `Cammoun et al., 2012 <https://
# www.ncbi.nlm.nih.gov/pubmed/22001222>`_). We're interested in assessing the
# degree to which these two vectors are correlated; that is, how does the
# strength of the structure-function relationship in a brain region relate to
# its position along the first diffusion gradient?

from scipy.stats import pearsonr

r, p = pearsonr(rsq, grad)
print('r = {:.2f}, p = {:.4g}'.format(r, p))

###############################################################################
# The p-value suggests that our data are, indeed, *highly* correlated.
# Unfortunately, when doing this sort of correlation with brain data the null
# model used to generate the p-value does not take into account that the data
# are constrained by a spatial toplogy (i.e., the brain) and are therefore
# spatially auto-correlated. The p-values will be "inflated" because our true
# degrees of freedom are less than the number of samples we have.
#
# To address this we can use a spatial permutation test (called a "spin test"),
# formally introduced by Alexander-Bloch et al., 2018, NeuroImage. This test
# works by considering the brain as a sphere and using random rotations of this
# sphere to construct a null distribution. If we rotate our data and resample
# datapoints based on their rotated values, we can generate a null that is more
# appropriate to our spatially auto-correlated data.
#
# The original spin test was designed for working with vertex-level data;
# however, since we have parcellated data there are a few different options we
# have to choose between when performing a spin test.
#
# Option 1: The "original" spin test
# ----------------------------------
#
# The original spin test assumes that you are working with two relatively high-
# resolution surface maps. It uses the coordinates of the vertices of these
# surfaces and applies random angular rotations, re-assigning values to the
# closest vertex (i.e., having the minimum Euclidean distance) after the
# rotation.
#
# However, there are instances when two vertices may be assigned the same value
# because their closest rotated vertex is identical. When working with surfaces
# that are sampled at a sufficiently high resolution this will occur less
# frequently, but does still happen with some frequency. To demonstrate we can
# grab the coordinates of the `fsaverage6` surface and generate a few
# rotations.
#
# First, we'll grab the spherical projections of the `fsaverage6` surface and
# extract the vertex coordinates:

import nibabel as nib

# if you have FreeSurfer installed on your computer this will simply grab the
# relevant files from the $SUBJECTS_DIR directory; otherwise, it will download
# them to the $HOME/nnt-data/tpl-fsaverage directory
lhsphere, rhsphere = nndata.fetch_fsaverage('fsaverage6', verbose=0)['sphere']

lhvert, lhface = nib.freesurfer.read_geometry(lhsphere)
rhvert, rhface = nib.freesurfer.read_geometry(rhsphere)

###############################################################################
# Then, we'll provide these to the function for generating the spin-based
# resampling array. We also need an indicator array designating which
# coordinates belong to which hemisphere so we'll create that first:

from netneurotools import stats as nnstats
import numpy as np

coords = np.vstack([lhvert, rhvert])
hemi = [0] * len(lhvert) + [1] * len(rhvert)
spins, cost = nnstats.gen_spinsamples(coords, hemi, n_rotate=10, seed=1234)
print(spins.shape)
print(spins)

###############################################################################
# ``spins`` is an array that contains the indices that we can use to resample
# the `fsaverage` surface according to ten random rotations. The ``cost`` array
# indicates the total cost (in terms of Euclidean distance) of the
# re-assignments for each rotation.
#
# The `fsaverage` surface has 81,924 vertices; let's check how many are
# re-assigned for each rotation and what the average re-assignment distance is
# for each vertex:

for rotation in range(10):
    uniq = len(np.unique(spins[:, rotation]))
    print('Rotation {:>2}: {} vertices, {:.2f} mm / vertex'
          .format(rotation + 1, uniq, cost[rotation] / len(spins)))

###############################################################################
# In this case we can see that, for the first rotation, only 75,380 vertices
# were re-assigned (meaning that we "lost" data from 6,544 vertices), but these
# were assigned to vertices that were, on average, about 0.67 mm from the
# original. While this doesn't seem too bad, when we lower the resolution of
# our data down even more (as we do with parcellations), this can become
# especially problematic.
#
# We can demonstrate this for the 1000-node parcellation that we have for our
# dataset above. We need to define the spatial coordinates of the parcels on
# a spherical surface projection. To do this, we'll fetch the left and right
# hemisphere FreeSurfer annotation files for the parcellation and then find the
# centroids of each parcel (defined on the spherical projection of the
# `fsaverage` surface):

from netneurotools import freesurfer as nnsurf

# this will download the Cammoun et al., 2012 FreeSurfer annotation files to
# the $HOME/nnt-data/atl-cammoun2012 directory
lhannot, rhannot = nndata.fetch_cammoun2012('surface', verbose=0)['scale500']

# this will find the center-of-mass of each parcel in the provided annotations
coords, hemi = nnsurf.find_fsaverage_centroids(lhannot, rhannot, surf='sphere')

###############################################################################
# The :func:`find_fsaverage_centroids` function return the xyz coordinates
# (``coords``) for each parcel defined in `lhannot` and `rhannot`, as well as
# an indicator array identifying to which hemisphere each parcel belongs
# (``hemi``):
#
# We'll use these coordinates to generate a resampling array as we did before
# for the `fsaverage6` vertex coordinates:

# we'll generate 1000 rotations here instead of only 10 as we did previously
spins, cost = nnstats.gen_spinsamples(coords, hemi, n_rotate=1000, seed=1234)
for rotation in range(10):
    uniq = len(np.unique(spins[:, rotation]))
    print('Rotation {:>2}: {} parcels, {:.2f} mm / parcel'
          .format(rotation + 1, uniq, cost[rotation] / len(spins)))

###############################################################################
# We can see two things from this: (1) we're getting more parcel duplications
# (i.e., only 727 out of the 1000 parcels were assigned in the first rotation),
# and (2) the distance from the original parcels to the re-assigned parcels has
# increased substantially from the `fsaverage6` data.
#
# This latter point makes sense: our parcellation provides a much sparser
# sampling of the cortical surface, so naturally parcels will be farther away
# from one another. However, this first issue of parcel re-assignment is a bit
# more problematic. At the vertex-level, when we're densely sampling the
# surface, it may not be as much of a problem that some vertices are
# re-assigned multiple times. But our parcels are a bit more "independent" and
# losing up to 300 parcels for each rotation may not be desirable.
#
# Nonetheless, we will use it to generate a spatial permutation-derived p-value
# for the correlation of our original data:

r, p = nnstats.permtest_pearsonr(rsq, grad, resamples=spins, seed=1234)
print('r = {:.2f}, p = {:.4g}'.format(r, p))

###############################################################################
# (Note that the maximum p-value from a permutation test is equal to ``1 /
# (n_perm + 1)``.)
#
# The benefit of generating our resampling array independent of a statistical
# test is that we can re-use it for any number of applications. If we wanted to
# conduct a Spearman correlation instead of a Pearson correlation we could
# easily do that:

from scipy.stats import rankdata

rho, prho = nnstats.permtest_pearsonr(rankdata(rsq), rankdata(grad),
                                      resamples=spins, seed=1234)
print('rho = {:.2f}, p = {:.4g}'.format(rho, prho))

###############################################################################
#
# Option 2: Exact matching
# ------------------------
##
# Option 3: Projection to vertex-space
# ------------------------------------
#
