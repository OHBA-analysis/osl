"""Preprocess CTF data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import preprocessing

# Settings
config = """
    preproc:
    - filter: {l_freq: 1, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}
    - bad_segments: {segment_len: 500, picks: mag}
    - bad_segments: {segment_len: 500, picks: mag, mode: diff}
    - bad_channels: {picks: mag}
    - interpolate_bads: {}
"""

# Create a list of paths to files to preprocess
inputs = [
    "data/raw/Nottingham/sub-not001/meg/sub-not001_task-resteyesopen_meg.ds",
    "data/raw/Nottingham/sub-not002/meg/sub-not002_task-resteyesopen_meg.ds",
]

# Directory to save output to
outdir = "data/preproc"

# Do preprocessing
preprocessing.run_proc_batch(
    config,
    inputs,
    outdir=outdir,
    overwrite=True,
)
