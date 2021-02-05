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
    pip install .

Alternatively, you can install ``netneurotools`` directly from PyPi with:

.. code-block:: bash

    pip install netneurotools

Optional installation for surface plotting
------------------------------------------

For surface plots like
:py:func:`netneurotools.plotting.plot_fsaverage`, you would need
VTK/mayavi/pysurfer installation. Here is a recipe for installing the
bleeding-edge version of the dependencies. If you already have a working
mayavi/pysurfer installation, there is generally no need to follow this
part.

-  Install Qt

   -  Installing Jupyterlab using conda (``conda install jupyterlab``)
      should automatically install the required qt/pyqt packages.
   -  If otherwise, using ``pip install PyQt5`` should also work as
      suggested by
      `here <http://docs.enthought.com/mayavi/mayavi/installation.html#latest-stable-release>`__
      and `here <https://github.com/enthought/mayavi#installation>`__
   -  If not working, search for the error prompts like
      `this <https://askubuntu.com/questions/308128/failed-to-load-platform-plugin-xcb-while-launching-qt5-app-on-linux-without>`__.

-  Install VTK

   -  Official wheels for the latest VTK9 are available for download
      `here <https://vtk.org/download/>`__.
   -  For Python>=3.9, official wheel is `not available at the
      moment <https://discourse.vtk.org/t/python-3-9/4369/3>`__.
      Following
      `here <https://docs.pyvista.org/extras/building_vtk.html>`__ and
      `here <https://gitlab.kitware.com/vtk/vtk/-/blob/master/Documentation/dev/build.md#python-wheels>`__,
      it’s possible to build the wheel. See the example code below.

.. code:: bash

   git clone https://github.com/Kitware/VTK
   cd VTK

   mkdir build
   cd build
   PYBIN=<PATH TO YOUR PYTHON EXECUTABLE>
   cmake -GNinja -DVTK_BUILD_TESTING=OFF -DVTK_WHEEL_BUILD=ON -DVTK_PYTHON_VERSION=3 -DVTK_WRAP_PYTHON=ON -DPython3_EXECUTABLE=$PYBIN ../

   # optionally, apt install ninja
   ninja
   $PYBIN setup.py bdist_wheel

   # to install
   pip install dist/vtk-*.whl

-  Install mayavi

   -  Install from source
      ``pip install git+https://github.com/enthought/mayavi.git``

-  Install pysurfer

   -  Install from source
      ``pip install git+https://github.com/nipy/PySurfer.git``

-  Install netneurotools

   -  Install from source
      ``pip install git+https://github.com/netneurolab/netneurotools.git``

Troubleshooting
~~~~~~~~~~~~~~~

-  Error related to ``from tvtk.vtk_module import VTK_MAJOR_VERSION``

   -  `Currently not
      fixed <https://github.com/enthought/mayavi/issues/939#issuecomment-747266625>`__
   -  Temporary workaround: adding ``VTK_MAJOR_VERSION = 9`` to
      ``mayavi/tvtk/vtk_module.py``

-  Error related to GLX

   -  Try ``glxgears`` or ``glxinfo``
   -  Check display driver compatibility
