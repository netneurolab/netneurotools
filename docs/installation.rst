.. _installation_setup:

Installation and setup
======================

.. _python_installation:

Python installation
-------------------

This package requires Python >= 3.6. Assuming you have the correct version of
Python installed, you can install ``netneurotools`` by opening a terminal and
running the following:

.. code-block:: bash

    git clone https://github.com/netneurolab/netneurotools.git
    cd netneurotools
    python setup.py install

Alternatively, you can install ``netneurotools`` directly from PyPi with:

.. code-block:: bash

    pip install netneurotools

.. _matlab_installation:

Matlab installation
-------------------

The Matlab functions should work on R2012B or later (though that is an estimate
based on the versions of Matlab that were around when the code was written).

To use the Matlab functions, download the repository and add the
`netneurotools_matlab` directory to your Matlab path.
