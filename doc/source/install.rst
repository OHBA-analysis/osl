Installing OSL
=================================


Install in Conda Environment
****************************

OSL can be installed in a conda environment using the following configuration. First save this into a textfile called ``osl-dev.yml``.

::

    name: osl
    channels:
    dependencies:
       - pip
       - pip:
         - git+https://github.com/OHBA-analysis/oslpy.git


Then run this command in your terminal

::

    conda env create -f osl-dev.yml

and finally:

::

    conda activate osl


Install from source code
************************

OSL can also be installed from source using git. This is the preferred developer option. First, change directory to the location you would like to install OSL in.

Next, run the following code in your terminal

::

    git clone https://github.com/OHBA-analysis/oslpy.git
    cd oslpy
    pip install .
