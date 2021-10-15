# OHBA MEG Quality Check

MNE-Python based tool for generating quality check reports from MEG recordings.

## Usage

These tools can be called from OSL in a script as follows.

```
osl.report.gen_report(list_of_files, outdir='/path/to/save/dir')
```

This function will generate a set of figures for each datafile in the input list, save them in the output directory and collate them into a html page for easy viewing.

You can optionally run some preprocessing on each datafile before the figures are generated. This might be useful to remove known artefacts (eg trimming breaks at the start and end of scan or filtering out HPI signal) so that the report can focus on the MEG signal. The preprocessing is specified using the same config format as the osl preprocessing tool. For example, here we trim the first 20 seconds of data and apply a broad filter to the data before reporting.

```
config = """
meta:
  event_codes:
preproc:
  - {method: crop, tmin: 20}
  - {method: filter, l_freq: 0.1, h_freq: 175}
"""

osl.report.gen_report(list_of_files, outdir='/path/to/save/dir', preproc_config=config)
```


## Command line usage

The script can also be run from the command line.

```
usage: osl_report [-h] [--outdir OUTDIR] [--config CONFIG] [--artefactscan]
                  files [files ...]

Run a quick Quality Control summary on data.

positional arguments:
  files            plain text file containing full paths to files to be
                   processed

optional arguments:
  -h, --help       show this help message and exit
  --outdir OUTDIR  Path to output directory to save data in
  --config CONFIG  yaml defining preproc
  --artefactscan   Generate additional plots assuming inputs are artefact
                   scans
```

for example:

```
osl_report my_list_of_meg_files.txt /full/path/to/my/output_directory
```

or you can use bash wildcards to identify files without first making a list in a text file

```
osl_report /path/to/my/data/resting_state*.fif /full/path/to/my/output_directory
```

