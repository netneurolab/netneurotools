.. _installation_setup:

Installation and setup
======================

.. _python_installation:

Python installation
-------------------

This package requires Python >= 3.5. Assuming you have the correct version of
Python installed, you can install ``netneurotools`` by opening a terminal and
running the following:

.. code-block:: bash

    git clone https://github.com/netneurolab/netneurotools.git
    cd netneurotools
    python setup.py install

Alternatively, you can install with ``pip``:

.. code-block:: bash

    git clone https://github.com/netneurolab/netneurotools.git
    pip install netneurotools

If you would prefer to use ``conda``, you can create a new environment using
the `environment.yml` file shipped with ``netneurotools``:

.. code-block:: bash

    git clone https://github.com/netneurolab/netneurotools.git
    conda env create -f netneurotools/environment.yml

.. note::

    The conda installation procedure is recommended for now; there are several
    optional plotting libraries that are very difficult to install with only
    ``pip`` but which conda handles elegantly. These libraries are thus listed
    as defaults in the ``environment.yml`` file but can only be installed with
    ``pip`` by calling ``pip install netneurotools[plotting]`` (n.b. that this
    command may fail!).

.. _matlab_installation:

Matlab installation
-------------------

The Matlab functions should work on R2012B or later (though that is an estimate
based on the versions of Matlab that were around when the code was written!).

To use the Matlab functions, download the repository and add the
`netneurotools_matlab` directory to your Matlab path.
