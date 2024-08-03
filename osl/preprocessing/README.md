# OSL Preprocessing tools

A python batch processing script for preprocessing of MEG files.

## Interactive Example

```
import osl

config = """
meta:
  event_codes:
    visual: 1
    auditory: 2
    button_press: 3
preproc:
  - crop:           {tmin: 40}
  - find_events:    {min_duration: 0.005}
  - filter:         {l_freq: 0.1, h_freq: 175}
  - notch_filter:   {freqs: 50 100 150}
  - bad_channels:   {picks: 'mag'}
  - bad_channels:   {picks: 'grad'}
  - bad_channels:   {picks: 'eeg'}
  - bad_segments:   {segment_len: 800, picks: 'mag'}
  - bad_segments:   {segment_len: 800, picks: 'grad'}
  - bad_segments:   {segment_len: 800, picks: 'eeg'}
  - resample:       {sfreq: 400, n_jobs: 6}
  - ica_raw:        {picks: 'meg', n_components: 64}
  - ica_autoreject: {picks: 'meg', ecgmethod: 'correlation'}
"""

# Output directory
outdir = '/where/do/i/want/my/output_dir'

# Process a single file
raw_file = '/path/to/file.fif'

osl.preprocessing.run_proc_chain(config, raw_file, outdir)  # creates /path/to/file_preproc-raw.fif

# Process a list of files
list_of_raw_files = ['/path/to/file1.fif','/path/to/file2.fif','/path/to/file3.fif']

osl.preprocessing.run_proc_batch(config, list_of_raw_files, outdir, overwrite=True)
```

### An example with epoching

```
config = """
meta:
  event_codes:
    visual: 1
    motor_short: 2
    motor_long: 3
preproc:
  - crop:              {tmin: 10}
  - set_channel_types: {EEG057: eog, EEG058: eog, EEG059: ecg}
  - pick_types:        {meg: true, eeg: false, eog: true,
                       ecg: true, stim: true, ref_meg: false}
  - find_events:       {min_duration: 0.005}
  - filter:            {l_freq: 1, h_freq: 175}
  - notch_filter:      {freqs: 50 100 150}
  - bad_channels:      {picks: 'meg'}
  - bad_segments:      {segment_len: 2000, picks: 'meg'}
  - epochs:            {tmin: -0.3, tmax: 1 }
  - tfr_multitaper:    {freqs: 4 45 41, n_jobs: 6, return_itc: false,
                        average: false, use_fft: false,
                        decim: 3, n_cycles: 2, time_bandwidth: 8}
"""
```

The following code runs the chain on a file:

```
fname = '/path/to/my/dataset.fif'

osl.preprocessing.run_proc_chain(config, fname)  # creates dataset_preproc-raw.fif and dataset_epo.fif

# Average the epochs object and visualise a response
epochs = mne.io.read_raw_fif('/path/to/my/dataset_epo.fif')
vis = epochs['visual'].average()
vis.plot_joint()
```

## Command Line Example

The command line function osl_preproc is installed with the package. This is a command line interface to run_proc_batch

```
Batch preprocess some fif files.

positional arguments:
  config                yaml defining preproc
  files                 plain text file containing full paths to files to be processed
  outdir                Path to output directory to save data in

optional arguments:
  -h, --help            show this help message and exit
  --logsdir LOGSDIR     Path to logs directory
  --reportdir REPORTDIR
                        Path to report directory
  --gen_report GEN_REPORT
                        Should we generate a report?
  --overwrite           Overwrite previous output files if they're in the way
  --verbose VERBOSE     Set the logging level for OSL functions
  --mneverbose MNEVERBOSE
                        Set the logging level for MNE functions
  --strictrun           Will ask the user for confirmation before starting
```

osl_preproc takes at least 2 arguments: `config` and `files, the rest are optional. For example:
```
osl_preproc my_config.yml list_of_raw_files.txt --outdir /path/to/my/output_dir --overwrite
```
