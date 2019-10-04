# -*- coding: utf-8 -*-
"""
Non-parametric significance testing with permutations
=====================================================

This example demonstrates how ``netneurotools`` can help perform null
hypothesis significance testing (NHST) via non-parametric permutation testing.
Many of the functions described here mirror functionality from the
``scipy.stats`` toolbox, but use non-parametric methods for deriving
significance (i.e., p-values).
"""

###############################################################################
# One-sample permutation tests
# ----------------------------
#
# Similar to a one-sample t-test, one-sample permutation tests are designed to
# estimate whether a group of values is different from some pre-specified null.
#
# First, we'll generate a random array with a mean and standard deviation of
# approximately five:

import numpy as np
np.random.seed(1234)
rvs = np.random.normal(loc=5, scale=5, size=(100, 2))

###############################################################################
# We can use ``scipy.stats`` for a standard parametric test to assess whether
# the array is different from zero:

from scipy import stats
print(stats.ttest_1samp(rvs, 0.0))

###############################################################################
# And can do the same thing with permutations using ``netneurotools.stats``:

from netneurotools import stats as nnstats
print(nnstats.permtest_1samp(rvs, 0.0))

###############################################################################
# Note that rather than returning a T-statistic with the p-values, the function
# returns the difference in means for each column from the null alongside the
# two-sided p-value.
#
# The :func:`~.permtest_1samp` function uses 1000 permutations to generate a
# null distribution. For each permutation it flips the signs of a random number
# of entries in the array and recomputes the difference in means from the null
# population mean. The return p-value assess the two-sided test of whether the
# absolute value of original difference in means is greater than the absolute
# value of the permuted differences.
#
# Just like with ``scipy``, we can test each column against an independent
# null mean:

print(nnstats.permtest_1samp(rvs, [5.0, 0.0]))

###############################################################################
# We can also provide an `axis` parameter (by default, `axis=0`):

print(nnstats.permtest_1samp(rvs.T, [5.0, 0.0], axis=1))

###############################################################################
# Finally, we can change the number of permutations we want to calculate (by
# default, `n_perm=1000`) and set a seed for reproducibility:

print(nnstats.permtest_1samp(rvs, 0.0, n_perm=500, seed=2222))

###############################################################################
# Note that the lowest p-value that can be obtained from a permutation test in
# ``netneurotools`` is equal to ``1 / (n_perm + 1)``.
#
# Two-sample related permutation tests
# ------------------------------------
#
# Similar to a two-sample t-test on related samples, two-sample related
# permutation tests are designed to estimate whether two groups of vlaues from
# the SAME SAMPLES are meaningfully different from one another.
#
# First, we'll generate two random arrays with means and standard deviations of
# approximately five, but one of the arrays will have a little bit of noise
# added to it:

rvs1 = np.random.normal(loc=5, scale=5, size=500)
rvs2 = (np.random.normal(loc=5, scale=5, size=500)
        + np.random.normal(scale=0.2, size=500))

###############################################################################
# These two arrays shouldn't be meaningfully different, and we can test that
# with a standard parametric test:

print(stats.ttest_rel(rvs1, rvs2))

###############################################################################
# Or with a non-parametric permutation test:

print(nnstats.permtest_rel(rvs1, rvs2))

###############################################################################
# To ensure that we're getting the results we'd expect, we can compare against
# a set of values that should be different:

rvs3 = (np.random.normal(loc=8, scale=10, size=500)
        + np.random.normal(scale=0.2, size=500))
print(nnstats.permtest_rel(rvs1, rvs3))

###############################################################################
# Permutation tests for correlations
# ----------------------------------
#
# Sometimes rather than assessing differences in means we want to assess the
# strength of a relationship between two variables. While we might normally do
# this with a Pearson (or Spearman) correlation, we can assess the significance
# of this relationship via permutation tests.
#
# First, we'll generate two correlated variables:

from netneurotools import datasets
x, y = datasets.make_correlated_xy(corr=0.2, size=100)

###############################################################################
# We can generate the Pearson correlation with the standard parametric p-value:

print(stats.pearsonr(x, y))

###############################################################################
# Or use permutation testing to derive the p-value:

print(nnstats.permtest_pearsonr(x, y))

###############################################################################
# All the same arguments as with :func:`~.permtest_1samp` and
# :func:`~.permtest_rel` apply here, so you can provide same-sized arrays and
# correlations will only be calculated for paired columns:

a, b = datasets.make_correlated_xy(corr=0.9, size=100)
arr1, arr2 = np.column_stack([x, a]), np.column_stack([y, b])
print(nnstats.permtest_pearsonr(arr1, arr2))

###############################################################################
# Or you can change the number of permutations and set a seed for
# reproducibility:

print(nnstats.permtest_pearsonr(arr1, arr2, n_perm=500, seed=2222))

###############################################################################
# Note that currently the `axis` parameter does not apply to
# :func:`~.permtest_pearsonr`.
