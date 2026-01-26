.. _installation_setup:

Installation and setup
======================

.. _python_installation:

Python installation
-------------------

This package requires Python >= 3.8. Assuming you have the correct version of
Python installed, you can install ``netneurotools`` by opening a terminal and
running the following:

.. code-block:: bash

    pip install git+https://github.com/netneurolab/netneurotools.git

Alternatively, you can install ``netneurotools`` directly from PyPi with:

.. code-block:: bash

    pip install netneurotools


Optional dependencies for surface plotting
------------------------------------------

Pyvista
~~~~~~~

This is the new plotting library used in the package. This will allow you to use functions like

-  :py:func:`netneurotools.plotting.pv_plot_surface`

You will need a working ``pyvista`` installation. 
Generally, we recommend using a clean conda environment, and install Pyvista using the following commands:

.. code-block:: bash

    conda create -n plotting python=3.12
    conda activate plotting
    conda install -c conda-forge pyvista
    # if you are using Jupyter notebooks
    conda install -c conda-forge jupyterlab trame trame-vtk trame-vuetify trame-jupyter-extension

Pyvista is actively maintained and generally installs without issues.
Please also check out their
`detailed installation guide <https://docs.pyvista.org/getting-started/installation.html>`_.

Pyvista is built on top of VTK, which supports multiple OpenGL window
backends and you can choose one that fits your environment
(see the `VTK runtime settings
<https://docs.vtk.org/en/latest/advanced/runtime_settings.html>`_):

- X11: standard Linux display server (requires an active X server)
- Win32: native Windows OpenGL context
- EGL: offscreen GPU context (good for headless servers with GPUs)
- OSMesa: CPU-based software rendering (good for headless servers without GPUs)

For example, on a headless Linux server without a GPU you can request OSMesa by
setting the backend before importing PyVista/VTK:

.. code-block:: python

    import os

    os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"



Pysurfer (deprecated)
~~~~~~~~~~~~~~~~~~~~~

This is the old plotting library used in the package. It is now deprecated in favor of Pyvista.
This will allow you to use functions like 

-  :py:func:`netneurotools.plotting.plot_fsaverage`
-  :py:func:`netneurotools.plotting.plot_fslr`
-  :py:func:`netneurotools.plotting.plot_conte69`
-  :py:func:`netneurotools.plotting.plot_fsvertex`

You will need a working ``vtk``/``mayavi``/``pysurfer`` installation.
These can generally be installed with the following command:

.. code-block:: bash

    pip install vtk mayavi pysurfer
