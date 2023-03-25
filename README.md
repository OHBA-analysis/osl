OHBA Software Library (OSL)
===========================

Install from Source Code
------------------------
The recommended installation depends on your operating system. OSL can be installed from source using:
```
git clone https://github.com/OHBA-analysis/osl.git
cd osl
conda env create -f envs/<os>.yml
conda activate osl
pip install -e .
```
where the environment file `<os>.yml` can be:

- `linux.yml` for a generic linux machine.
- `m1_mac.yml` if you are using a modern Mac computer.
- `hbaws.yml` if you are using an OHBA workstation at Oxford.
- `bmrc.yml` if you are using the BMRC at Oxford.

Note, all of the above environments come with Jupyter Notebook installed. The `hbaws.yml` and `m1_mac.yml` environments also comes with Spyder installed.

Deleting osl
------------
If you installed osl using the instructions above then to completely remove it simply delete the conda environment and delete the repo on your local machine:
```
conda env remove -n osl
rm -rf osl
```

For Developers
--------------
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
