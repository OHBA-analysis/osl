"""Preprocessing.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import preprocessing

raw_file = "/ohba/pi/knobre/datasets/covid/rawbids/{0}/meg/{0}_task-restEO.fif"
preproc_dir = "/ohba/pi/knobre/cgohil/covid/preproc"

subjects = ["sub-006", "sub-007"]

# Settings
config = """
    preproc:
    - filter: {l_freq: 0.5, h_freq: 125, method: 'iir', iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}
    - bad_channels: {picks: mag}
    - bad_channels: {picks: grad}    
    - bad_segments: {segment_len: 2000, picks: mag}
    - bad_segments: {segment_len: 2000, picks: grad}
    - ica_raw: {n_components: 20, picks: meg}
    - ica_autoreject: {apply: False}
"""

# Input files
inputs = [raw_file.format(subject) for subject in subjects]

# Preprocess
preprocessing.run_proc_batch(
    config,
    inputs,
    outdir=preproc_dir,
    overwrite=True,
)
