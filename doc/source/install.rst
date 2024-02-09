Installation
============

Conda
-----

We recommend installing OSL within a virtual environment. You can do this with `Anaconda <https://docs.anaconda.com/free/anaconda/install/index.html>`_ (or `miniconda <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_).

Linux
-----

1. Install FSL using the instructions [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux).

2. Install OSL either via pip using::

    pip install osl

or from source (in editable mode) using::

    git clone https://github.com/OHBA-analysis/osl.git
    cd osl
    conda env create -f envs/linux.yml
    conda activate osl
    pip install -e .

Windows
-------

If you're using a Windows machine, installing [Ubuntu](https://ubuntu.com/wsl) (linux) using a Windows subsystem. We recommend following the FSL instructions [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Windows) then installing OSL using the instructions above.
