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
  - {method: ica_raw, picks: 'meg', n_components: 64}
  - {method: ica_autoreject, picks: 'meg', ecgmethod: 'correlation'}
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

### Load config from file

If your config is saved to disk in a text file, you can load it in like this

```
config = osl.preprocessing.check_inconfig('/path/to/my/config.yaml')
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
  - {method: crop, tmin: 10}
  - {method: set_channel_types, EEG057: eog, EEG058: eog, EEG059: ecg}
  - {method: pick_types, meg: true, eeg: false, eog: true,
                         ecg: true, stim: true, ref_meg: false}
  - {method: find_events, min_duration: 0.005}
  - {method: filter, l_freq: 1, h_freq: 175}
  - {method: notch_filter, freqs: 50 100 150}
  - {method: bad_channels, picks: 'meg'}
  - {method: bad_segments, segment_len: 2000, picks: 'meg'}
  - {method: epochs, tmin: -0.3, tmax: 1 }
  - {method: tfr_multitaper, freqs: 4 45 41, n_jobs: 6, return_itc: false,
                             average: false, use_fft: false,
                             decim: 3, n_cycles: 2, time_bandwidth: 8}
```

and the following code runs the chain on a file:

```
config = osl.preprocessing.check_inconfig('myconfig.yaml`)

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
