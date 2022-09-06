OHBA Software Library (OSL) in Python
=====================================
Python version of https://github.com/OHBA-analysis/osl-core.

Installation
------------
The recommended installation depends on your operating system. If you are installing on a Mac or Linux machine, we recommend:
```
git clone https://github.com/OHBA-analysis/oslpy.git
cd oslpy
conda env create -f envs/linux.yml
conda activate osl
pip install -e .
```
Here, the `-e` indicates we have installed in 'editable mode'. This means if we execute `git pull` or make any local modifications these changes will be reflected when we import the package.

If you use the OHBA workstation (hbaws) with TigerVNC, we recommend:
```
git clone https://github.com/OHBA-analysis/oslpy.git
cd oslpy
conda env create -f envs/hbaws-vnc.yml
conda activate osl
pip install -e .
```

On remote servers without displays specific package versions are required (related to source reconstruction visualisation). The following can be used to install on machines without a display:
```
git clone https://github.com/OHBA-analysis/oslpy.git
cd oslpy
conda env create -f envs/headless-server.yml
conda activate osl
pip install -e .
```

If you use the BMRC server, we recommend:
```
git clone https://github.com/OHBA-analysis/oslpy.git
cd oslpy
conda env create -f envs/bmrc.yml
conda activate osl
pip install -e .
```

If you use the OHBA workstation (hbaws) via the terminal only, we recommend:
```
git clone https://github.com/OHBA-analysis/oslpy.git
cd oslpy
conda env create -f envs/hbaws-no-display.yml
conda activate osl-nd
pip install -e .
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
