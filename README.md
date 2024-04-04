# OHBA Software Library (OSL)

Tools for MEG/EEG analysis.

Documentation: https://osl.readthedocs.io/en/latest/.

## Installation

See the [official documentation](https://osl.readthedocs.io/en/latest/install.html) for recommended installation instructions.

Alternatively, OSL can be installed from source code within a [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) (or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)) environment using the following.

### Linux

```
git clone https://github.com/OHBA-analysis/osl.git
cd osl
conda env create -f envs/linux.yml
conda activate osl
pip install -e .
```

### Mac

```
git clone https://github.com/OHBA-analysis/osl.git
cd osl
conda env create -f envs/mac.yml
conda activate osl
pip install -e .
```

## Removing OSL

Simply removing the conda environment and delete the repository:
```
conda env remove -n osl
rm -rf osl
```

## For Developers

Run tests:
```
cd osl
pytest tests
```
or to run a specific test:
```
cd osl/tests
pytest test_file_handling.py
```

Build documentation:
```
python setup.py build_sphinx
```
Compiled docs can be found in `doc/build/html/index.html`.
