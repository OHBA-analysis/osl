"""Example script for preprocessing CamCAN in serial.

In this script the preprocessing will be done for one fif file at a time.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import pathlib
from glob import glob

from osl import preprocessing

rawdir = "/ohba/pi/mwoolrich/datasets/CamCan_2021/cc700/meg/pipeline/release005/BIDSsep/rest"
outdir = "/ohba/pi/mwoolrich/cgohil/camcan/preproc"

config = """
    preproc:
    - filter: {l_freq: 0.5, h_freq: 125, method: 'iir', iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100 150 200}
    - resample: {sfreq: 250}
    - bad_channels: {picks: 'mag'}
    - bad_channels: {picks: 'grad'}        
    - bad_segments: {segment_len: 2000, picks: 'mag'}
    - bad_segments: {segment_len: 2000, picks: 'grad'}
"""

# Get input files
inputs = []
for subject in sorted(glob(f"{rawdir}/sub-*")):
    subject = pathlib.Path(subject).stem
    inputs.append(f"{rawdir}/{subject}/ses-rest/meg/{subject}_ses-rest_task-rest_meg.fif")
inputs = inputs[:2]

# Main preprocessing
preprocessing.run_proc_batch(
    config,
    inputs,
    outdir=outdir,
    overwrite=True,
)
