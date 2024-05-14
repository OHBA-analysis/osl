# Conda Environments

OSL only environments:

- `linux.yml`: for Linux computers.
- `mac.yml`: for Mac computers.
- `hbaws.yml`: for Oxford OHBA workstation computers.
- `bmrc.yml`: for the Oxford BMRC cluster.

These can be install with:
```
git clone https://github.com/OHBA-analysis/osl.git
cd osl
conda env create -f envs/<os>.yml
conda activate osl
pip install -e .
```

OSL + osl-dynamics environments:

- **`linux-full.yml`: recommended environment for Linux computers.**
- **`linux-full-with-spyder.yml`: full installation including spyder.
- **`mac-full.yml`: recommended environment for Mac computers.**
- `osl-workshop-23.yml`: used in the [2023 OSL Workshop](https://osf.io/zxb6c/).

See the official documentation for the [installation instructions](https://osl.readthedocs.io/en/latest/install.html) for these environment files.

All environments come with Jupyter Notebook.
