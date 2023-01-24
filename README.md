OHBA Software Library (OSL)
===========================

Install from Source Code
------------------------
The recommended installation depends on your operating system. If you are installing on a Mac or Linux machine, we recommend:
```
git clone https://github.com/OHBA-analysis/osl.git
cd osl
conda env create -f envs/linux.yml
conda activate osl
pip install -e .
```
Here, the `-e` indicates we have installed in 'editable mode'. This means if we execute `git pull` or make any local modifications these changes will be reflected when we import the package.

If you use the OHBA workstation (hbaws), we recommend:
```
git clone https://github.com/OHBA-analysis/osl.git
cd osl
conda env create -f envs/hbaws.yml
conda activate osl
pip install -e .
```
The hbaws environment comes with jupyter notebook already installed. If you'd like to use spyder then you can install this with
```
conda activate osl
pip install spyder==5.1.5
```
Spyder version 5.1.5 is needed if you're using CentOS 7. If you are using a newer hbaws machine (one with Rocky linux) then you don't need to specify the version. Note, you might launch a version of spyder installed in your base conda environment even though you've activated the osl environment. If so, you might have problems importing packages from osl. To fix this you can uninstall spyder from your base environment.

If you use the BMRC server, we recommend:
```
git clone https://github.com/OHBA-analysis/osl.git
cd osl
conda env create -f envs/bmrc.yml
conda activate osl
pip install -e .
```

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
