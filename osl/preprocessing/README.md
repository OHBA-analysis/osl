# OSL Preprocessing tools

A python batch processing script for preprocessing of MEG files.

## Interactive Example

```
import mne
import osl
import yaml

config_text = """
meta:
  event_codes:
    visual: 1
    auditory: 2
    button_press: 3
preproc:
  - {method: crop, tmin: 40}
  - {method: find_events, min_duration: 0.005}
  - {method: filter, l_freq: 0.1, h_freq: 175}
  - {method: notch_filter, freqs: 50 100 150}
  - {method: bad_channels, picks: 'mag'}
  - {method: bad_channels, picks: 'grad'}
  - {method: bad_channels, picks: 'eeg'}
  - {method: bad_segments, segment_len: 800, picks: 'mag'}
  - {method: bad_segments, segment_len: 800, picks: 'grad'}
  - {method: bad_segments, segment_len: 800, picks: 'eeg'}
  - {method: resample, sfreq: 400, n_jobs: 6}
  - {method: ica_raw_autoreject, picks: 'meg', ecgmethod: 'correlation'}
"""

config = yaml.load(config_text, Loader=yaml.FullLoader)
raw = mne.io.read_raw_fif('/path/to/my/raw_data.fif', preload=True)
outdir = '/where/do/i/want/my/output_dir'

# Process a single file
dataset = osl.preprocessing.run_proc_chain(raw, config)

# Process a list of files
list_of_raw_files = ['/path/to/file1.fif','/path/to/file2.fif','/path/to/file3.fif']
osl.preprocessing.run_proc_batch(config, list_of_raw_files, outdir, overwrite=True)
```


## Command Line Example

The command line function osl_batch is installed with the package. This is a command line interface to run_proc_batch

```
usage: usage: osl_batch [-h] [--overwrite] config files outdir

Batch preprocess some fif files.

positional arguments:
  config       yaml defining preproc
  files        plain text file containing full paths to files to be processed
  outdir       Path to output directory to save data in

optional arguments:
  -h, --help   show this help message and exit
  --overwrite  Overwrite previous output files if they're in the way
```

for example...

```
osl_batch my_config.yaml list_of_raw_files.txt /path/to/my/output_dir --overwrite
