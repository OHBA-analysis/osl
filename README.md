OHBA Software Library (OSL) in Python
=====================================
Python version of https://github.com/OHBA-analysis/osl-core.

Installation
------------
The recommended installation is:
```
git clone https://github.com/OHBA-analysis/oslpy.git
cd oslpy
conda env create -f envs/osl.yml
conda activate osl
```

On remote servers without displays specific package versions are required (related to source reconstruction visualisation). The following environment if recommended if you are installing on the OHBA workstation.
```
conda env create -f envs/hbaws.yml
```

For Developers
--------------
Install in editable mode:
```
conda create --name osl python=3
conda activate osl
git clone git@github.com:OHBA-analysis/oslpy.git
cd oslpy
pip install -e .
```

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
