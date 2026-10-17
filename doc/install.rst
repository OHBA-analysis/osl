Installation
============

osl-ephys is available as a conda package on the `OHBA channel <https://anaconda.org/ohba-analysis-group>`_ of anaconda.org and on `PyPI <https://pypi.org/project/osl-ephys>`_.

Source reconstruction also needs `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_ for extracting the head surfaces from a structural MRI, and the surface-based route needs `FreeSurfer <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_. Neither is needed for preprocessing. On Windows, install everything inside `Ubuntu on WSL <https://ubuntu.com/wsl>`_.

Conda installation (recommended)
--------------------------------

We recommend installing osl-ephys with `Miniforge <https://conda-forge.org/download/>`_, which provides :code:`conda` and :code:`mamba`. If you do not already have it:

.. code::

    curl -LO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    rm Miniforge3-$(uname)-$(uname -m).sh

Then create an environment with osl-ephys in it:

.. code::

    mamba create -n osle -c ohba-analysis-group -c conda-forge osl-ephys
    conda activate osle

This installs osl-ephys from the OHBA channel with everything it needs from conda-forge, including `MNE-Python <https://mne.tools/stable/index.html>`_.

glmtools and sails, which the GLM statistics module uses, are not on conda-forge; the OHBA channel carries them too, so they are installed along with the rest. (A pip installation gets them from PyPI.)

On a headless Linux server, set :code:`PYVISTA_OFF_SCREEN=true` so the 3D coregistration plots render without a display; the package brings the software-rendering build of VTK with it there.

Pip installation
----------------

osl-ephys can be installed from PyPI with:

.. code::

    pip install osl-ephys

Note that pip will not install the non-Python libraries that some dependencies need, which is why we recommend conda.

Install the latest development code (optional)
----------------------------------------------

You should only need to do this if you need a feature or fix that has not been released yet.

Once you have created the :code:`osle` conda environment (see instructions above) you can install the latest development version on the `GitHub repository <https://github.com/OHBA-analysis/osl-ephys>`_ with:

.. code::

    conda activate osle
    pip install git+https://github.com/OHBA-analysis/osl-ephys.git

Copy the source code (optional)
-------------------------------

Once you have created the :code:`osle` conda environment (see instructions above) you can install a local copy of the source code (`GitHub repository <https://github.com/OHBA-analysis/osl-ephys>`_) into it:

.. code::

    git clone https://github.com/OHBA-analysis/osl-ephys.git
    conda activate osle
    cd osl-ephys
    pip install -e .

Now you can run osl-ephys with your own local changes to the code.

Removing osl-ephys
------------------

To remove osl-ephys simply delete the conda environment:

::

    conda env remove -n osle
    conda clean --all

The downloaded data files are cached separately, and can be removed with:

::

    rm -rf ~/.cache/osl-files        # linux
    rm -rf ~/Library/Caches/osl-files  # macOS
