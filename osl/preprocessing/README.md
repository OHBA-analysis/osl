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

config = osl.preprocessing.load_config(config_text)
infile = '/path/to/my/raw_data.fif'
outdir = '/where/do/i/want/my/output_dir'

# Process a single file from disk
dataset = osl.preprocessing.run_proc_chain(infile, config)

# Process a single file from the workspace
raw = mne.io.read_raw_fif('/path/to/my/raw_data.fif', preload=True)
dataset = osl.preprocessing.run_proc_chain(raw, config)

# Process a list of files
list_of_raw_files = ['/path/to/file1.fif','/path/to/file2.fif','/path/to/file3.fif']
osl.preprocessing.run_proc_batch(config, list_of_raw_files, outdir, overwrite=True)
```

### Load config from file

If your config is saved to disk in a text file, you can load it in like this

```
config = osl.preprocessing.load_config('/path/to/my/config.yaml')
```

### An example with epoching

This config is saved in a text file called myconfig.yaml

```
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
```

and the following code runs the chain on a file:

```
config = osl.preprocessing.load_config('myconfig.yaml`)

fname = '/path/to/my/dataset.fif'

dataset = osl.preprocessing.run_proc_chain(fname, config)

# Average the epochs object and visualise a response
vis = dataset['epochs']['visual'].average()
vis.plot_joint()
```

### Editing the config

Once loaded, the config is a python dictionary which can be viewed and edited online.

```
# view first stage
print(config['preproc'][0])

# view second stage
print(config['preproc'][1])

# Change filter parameters
config['preproc'][4]['l_freq'] = 0.1

# Remove TFR stage
del config['preproc'][-1]

print(config['preproc'])
```

You can then run the edited config as normal

```
dataset = osl.preprocessing.run_proc_chain(fname, config)
```

## Command Line Example

The command line function osl_batch is installed with the package. This is a command line interface to run_proc_batch

```
usage: usage: osl_batch [-h] [--overwrite] [--nprocesses NPROCESSES]
                 [--mnelog MNELOG]
                 config files outdir

Batch preprocess some fif files.

positional arguments:
  config                yaml defining preproc
  files                 plain text file containing full paths to files to be
                        processed
  outdir                Path to output directory to save data in

optional arguments:
  -h, --help            show this help message and exit
  --overwrite           Overwrite previous output files if they're in the way
  --nprocesses NPROCESSES
                        Number of jobs to process in parallel
  --mnelog MNELOG       Set the logging level for MNE python functions

```

for example...

```
osl_batch my_config.yaml list_of_raw_files.txt /path/to/my/output_dir --overwrite
```

## Available processing options

| Name              | Base Module | Description |
| ---               |  ---          |  ---- |
| anonymize         | mne.io.Raw       | Anonymize measurement information in place. |
| apply_hilbert     | mne.io.Raw       | Compute analytic signal or envelope for a subset of channels. |
| crop              | mne.io.Raw       | Crop raw data file. |
| drop_channels     | mne.io.Raw       | Drop channel(s). |
| filter            | mne.io.Raw       | Filter a subset of channels. |
| interpolate_bads  | mne.io.Raw       | Interpolate bad MEG and EEG channels. |
| notch_filter      | mne.io.Raw       | Notch filter a subset of channels. |
| pick_channels     | mne.io.Raw       | Pick some channels. |
| pick_types        | mne.io.Raw       | Pick some channels by type and names. |
| rename_channels   | mne.io.Raw       | Rename channels. |
| resample          | mne.io.Raw       | Resample all channels. |
| savgol_filter     | mne.io.Raw       | Filter the data using Savitzky-Golay polynomial method. |
| set_channel_types | mne.io.Raw       | Define the sensor type of channels. |
| set_eeg_reference | mne.io.Raw       | Specify which reference to use for EEG data. |
| set_meas_date     | mne.io.Raw       | Set the measurement start date. |
| drop_bad          | mne.Epochs    | Drop bad epochs without retaining the epochs data. |
| apply_baseline    | mne.Epochs    | Baseline correct epochs. |
| annotate_flat     | mne.preprocessing | Annotate flat segments of raw data (or add to a bad channel list). |
| annoatate_muscle_zscore | mne.preprocessing | Create annotations for segments that likely contain muscle artifacts. |
| compute_current_source_density | mne.preprocessing | Get the current source density (CSD) transformation. |
| tfr_multitaper | mne.time_frequency | Compute Time-Frequency Representation (TFR) using DPSS tapers. |
| tfr_morlet     | mne.time_frequency | Compute Time-Frequency Representation (TFR) using Morlet wavelets. |
| sft_stockwell  | mne.time_frequency | Compute Time-Frequency Representation (TFR) using Stockwell Transform. |
| bad_channels   | osl.preprocessing  | Detect bad channels using the GESD Algorithm. |
| bad_segments   | osl.preprocessing  | Annotate bad segments using the GESD Algorithm. |
