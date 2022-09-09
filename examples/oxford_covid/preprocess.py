"""Preprocessing.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import preprocessing

RAW_FILE = "/ohba/pi/knobre/datasets/covid/rawbids/sub-{0}/meg/sub-{0}_task-restEO.fif"
PREPROC_DIR = "/ohba/pi/knobre/cgohil/covid/preproc"

SUBJECTS = ["004", "005"]

# Settings
config = """
    preproc:
    - filter: {l_freq: 0.5, h_freq: 125, method: 'iir', iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}
    - bad_segments: {segment_len: 2000, picks: 'mag'}
    - bad_segments: {segment_len: 2000, picks: 'grad'}
    - bad_channels: {picks: 'meg'}
    - ica_raw: {n_components: 20, picks: 'meg'}
    - ica_autoreject: {apply: False}
"""

# Setup
inputs = []
for subject in SUBJECTS:
    inputs.append(RAW_FILE.format(subject))

# Preprocess
preprocessing.run_proc_batch(
    config,
    inputs,
    outdir=PREPROC_DIR,
    overwrite=True,
)
