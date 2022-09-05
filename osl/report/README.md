# OHBA MEG Quality Check

MNE-Python based tool for generating quality check reports from MEG recordings.

## Usage

These tools can be called from OSL in a script as follows.

```
osl.report.gen_report_from_fif(list_of_files, outdir='/path/to/save/dir')
```

This function will generate a set of figures for each data file in the input list, save them in the output directory and collate them into a html page for easy viewing.


## Command line usage

The script can also be run from the command line.

```
usage: osl_report [-h] files [files ...] outdir [OUTDIR]

Run a quick quality control summary on data.

positional arguments:
  files          plain text file containing full paths to files to be processed
  outdir OUTDIR  Path to output directory to save data in

optional arguments:
  -h, --help       show this help message and exit
```
