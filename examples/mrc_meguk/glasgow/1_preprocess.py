"""Preprocesses data from the Glasgow site in the MRC MEGUK Partnership dataset.

"""

from osl.preprocessing import run_proc_batch
from glob import glob

# Authors : Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

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
      - bad_segments: {segment_len: 2000, picks: 'mag'}
      - resample: {sfreq: 400, npad: 'auto'}
      - ica_raw: {n_components: 60, picks: 'meg'}
"""

# Raw data filenames
raw_data_files = []
raw_data_files += sorted(glob("/ohba/pi/mwoolrich/datasets/mrc_meguk_public/Glasgow/*/meg/*resteyesclosed*/c,rfDC"))
raw_data_files += sorted(glob("/ohba/pi/mwoolrich/datasets/mrc_meguk_public/Glasgow/*/meg/*resteyesopen*/c,rfDC"))
raw_data_files += sorted(glob("/ohba/pi/mwoolrich/datasets/mrc_meguk_public/Glasgow/*/meg/*visuomotor*/c,rfDC"))

# Directory for preprocessed data
output_dir = "/ohba/pi/mwoolrich/cgohil/mrc_meguk_public/Glasgow"

run_proc_batch(config, raw_data_files, output_dir, nprocesses=4)
