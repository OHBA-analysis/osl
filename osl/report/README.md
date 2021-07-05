# OHBA MEG Quality Check

MNE-Python based tool for generating quality check reports from MEG recordings.

## Install

This tool can be run as a stand alone script without direct installation but the dependancies in the requirements.txt file must be installed in the present environment. These can be installed using pip

```
pip install -r requirements.txt
```

## Usage

The script runs from the command line with two positional arguments

```
positional arguments:
  files       plain text file containing full paths to files to be processed
  outdir      Path to output directory to save data in

optional arguments:
  -h, --help  show this help message and exit
```

for example:

```
python ohba_meg_qc.py my_list_of_meg_files.txt /full/path/to/my/output_directory
```
