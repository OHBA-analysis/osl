Installation
============

First, make sure you have conda (or miniconda) installed: https://docs.conda.io/en/latest. Then oslpy can be installed from source using the following:

::
    
    git clone https://github.com/OHBA-analysis/oslpy.git
    cd oslpy
    conda env create -f envs/linux.yml
    conda activate osl
    pip install -e .
