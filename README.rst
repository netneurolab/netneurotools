netneurotools: Tools for network neuroscience
=============================================

|

.. .. image:: https://zenodo.org/badge/375755159.svg
..    :target: https://zenodo.org/badge/latestdoi/375755159
..    :alt: Zenodo record

.. image:: https://img.shields.io/pypi/v/netneurotools
   :target: https://pypi.python.org/pypi/netneurotools/
   :alt: Latest PyPI version

.. image:: https://img.shields.io/badge/docker-netneurolab/netneurotools-brightgreen.svg?logo=docker&style=flat
  :target: https://hub.docker.com/r/netneurolab/netneurotools/tags/
  :alt: Latest Docker image

.. image:: https://github.com/netneurolab/netneurotools/actions/workflows/tests.yml/badge.svg
  :target: https://github.com/netneurolab/netneurotools/actions/workflows/tests.yml
  :alt: run-tests status

.. image:: https://github.com/netneurolab/netneurotools/actions/workflows/docs.yml/badge.svg
  :target: https://netneurolab.github.io/netneurotools/
  :alt: deploy-docs status

|

This toolbox is a collection of functions written in Python that get frequent
usage in the `Network Neuroscience Lab <netneurolab.github.io/>`_, housed in
the `Brain Imaging Centre <https://www.mcgill.ca/bic/home>`_ at
`McGill University <https://www.mcgill.ca/>`_.

.. _installation:

Installation
------------

Install directly from PyPi with :code:`pip install netneurotools` or install the main branch with

.. code-block:: bash

    git clone https://github.com/netneurolab/netneurotools.git
    cd netneurotools
    pip install .

.. _features:

Features
--------

*  Network neuroscience metrics: up-to-date and optimized

   *  Network communication
   *  Null networks

*  Brain plotting functions: easy to use and customize

   *  Surface visualization 
      `plot_fsaverage <https://netneurotools.readthedocs.io/en/latest/generated/netneurotools.plotting.plot_fsaverage.html>`_ 
      and `plot_fslr <https://netneurotools.readthedocs.io/en/latest/generated/netneurotools.plotting.plot_fslr.html>`_
   *  3D point brain `plot_point_brain <https://netneurotools.readthedocs.io/en/latest/generated/netneurotools.plotting.plot_point_brain.html>`_
   *  Sorted communities `plot_mod_heatmap <https://netneurotools.readthedocs.io/en/latest/generated/netneurotools.plotting.plot_mod_heatmap.html>`_

*  Statistics functions

   *  Dominance analysis `get_dominance_stats <https://netneurotools.readthedocs.io/en/latest/generated/netneurotools.stats.get_dominance_stats.html>`_

*  Fetchers for common datasets

*  Utilities for working with FreeSurfer and CIVET


Check out our `documentation <https://netneurotools.readthedocs.io/en/latest>`_
for more information!

.. _development:

Development
-----------

This package has been developed by members of the Network Neuroscience Lab in
pursuit of their research. While we've made every effort to ensure these tools
work and have some documentation, there is always room for improvement! If
you've found a bug, are experiencing a problem, or have a question, create a
new `issue <https://github.com/netneurolab/netneurotools/issues>`_ with some
information about it and one of our team members will do our best to help you.

.. _licensing:

License Information
-------------------

This codebase is licensed under the 3-clause BSD license. The full license can
be found in the `LICENSE <https://github.com/netneurolab/netneurotools/blob/
master/LICENSE>`_ file in the ``netneurotools`` distribution.

All trademarks referenced herein are property of their respective holders.

.. |sparkles| replace:: ✨
