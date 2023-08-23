"""Preprocesses data from the Aston site in the MRC MEGUK Partnership dataset.

Note: there is no EOG/ECG from this site.
"""

# Authors : Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl.preprocessing import run_proc_batch
from glob import glob

# Settings
config = """
    meta:
      event_codes:
        visual: 30
        motor_short: 31
        motor_long: 32
    preproc:
      - crop: {tmin: 10}
      - find_events: {min_duration: 0.005}
      - filter: {l_freq: 0.1, h_freq: 175}
      - notch_filter: {freqs: 50 100 150}
      - bad_channels: {picks: 'mag'}
      - bad_channels: {picks: 'grad'}
      - bad_segments: {segment_len: 2000, picks: 'mag'}
      - bad_segments: {segment_len: 2000, picks: 'grad'}
      - resample: {sfreq: 400, npad: 'auto'}
"""

# Raw data filenames
raw_data_files = []
raw_data_files += sorted(glob("/ohba/pi/mwoolrich/datasets/mrc_meguk_public/Aston/derivatives/*/meg/*resteyesclosed*.fif"))
raw_data_files += sorted(glob("/ohba/pi/mwoolrich/datasets/mrc_meguk_public/Aston/derivatives/*/meg/*resteyesopen*/*.fif"))
raw_data_files += sorted(glob("/ohba/pi/mwoolrich/datasets/mrc_meguk_public/Aston/derivatives/*/meg/*visuomotor*/*.fif"))

# Directory for preprocessed data
output_dir = "/ohba/pi/mwoolrich/cgohil/mrc_meguk_public/Aston"

run_proc_batch(config, raw_data_files, output_dir, nprocesses=4)
