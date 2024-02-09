Installation
============

We recommend installing OSL within a virtual environment. You can do this with `Anaconda <https://docs.anaconda.com/free/anaconda/install/index.html>`_ (or `miniconda <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_).

Linux
-----

1. Install FSL using the instructions `here <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux>`_.

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

If you're using a Windows machine, we recommend you install FSL within a `Ubuntu <https://ubuntu.com/wsl>`_ (linux) subsystem following the instructions `here <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Windows>`_.

Then install OSL using the instructions above.
