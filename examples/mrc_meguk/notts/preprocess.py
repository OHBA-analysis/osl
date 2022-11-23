"""Preprocessing.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>

from glob import glob
from pathlib import Path

from osl import preprocessing

RAW_DIR = "/ohba/pi/mwoolrich/datasets/mrc_meguk_public/Nottingham"
RAW_FILE = RAW_DIR + "/{0}/meg/{0}_task-resteyesopen_meg.ds"
PREPROC_DIR = "/ohba/pi/mwoolrich/cgohil/ukmp_notts/preproc"

# Settings

# Note that with CTF, MNE will return:
# ~274 axial grads (as magnetometers) if {picks: mag, ref_meg: False}
# ~28 reference axial grads if {picks: grad}

config = """
    preproc:
    - set_channel_types: {EEG057: eog, EEG058: eog, EEG059: ecg}
    - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}
    - crop: {tmin: 10, tmax: 300}           
    - set_channel_types: {EEG057: eog, EEG058: eog, EEG059: ecg}
    - bad_channels: {picks: mag, ref_meg: False, significance_level: 0.1}
    - bad_channels: {picks: grad, significance_level: 0.4}
    - bad_segments: {segment_len: 600, picks: mag, ref_meg: False, significance_level: 0.1}
"""

# Setup
inputs = []
for directory in sorted(glob(RAW_DIR + "/sub-*")):
    subject = Path(directory).name
    inputs.append(RAW_FILE.format(subject))
inputs = inputs[:2]

# Preprocess
preprocessing.run_proc_batch(
    config,
    inputs,
    outdir=PREPROC_DIR,
    overwrite=True,
)
