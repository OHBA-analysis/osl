Installing oslpy
================

Currently, oslpy can only be installed from source. We recommend installing in a conda environment (described below).

Conda Installation (Recommended)
********************************

First, make sure you have conda (or miniconda) installed: https://docs.conda.io/en/latest. Then oslpy can be installed from source using the following:

::
    
    git clone https://github.com/OHBA-analysis/oslpy.git
    cd oslpy
    conda env create -f envs/osl.yml
    conda activate osl


PIP Installation
****************

Alternatively, oslpy can be installed in your existing virtual environment with:

::

    git clone https://github.com/OHBA-analysis/oslpy.git
    cd oslpy
    pip install .

Developers will want to install in editable mode:

::

    pip install -e .
