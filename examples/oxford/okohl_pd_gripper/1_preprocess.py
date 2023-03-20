"""Preprocess MaxFiltered sensor data.

"""

# Authors: Oliver Kohl <oliver.kohl@psych.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import preprocessing

# Subjects to preprocess
subjects = ["HC01", "HC02"]

# Settings
config = """
    preproc:
    - notch_filter: {freqs: 50 100 150 200 250}
    - filter: {l_freq: 1, h_freq: 45, fir_design: firwin}
    - resample: {sfreq: 250, npad: auto}
    - bad_segments: {segment_len: 1000, picks: mag}
    - bad_segments: {segment_len: 1000, picks: grad}
    - bad_segments: {segment_len: 1000, picks: mag, mode: diff}
    - bad_segments: {segment_len: 1000, picks: grad, mode: diff}
    - ica_raw: {n_components: 0.99, picks: meg}
    - ica_autoreject: {apply: False}
"""

# Setup paths to maxfiltered data files
inputs = []
for subject in subjects:
    inputs.append(
        f"/ohba/pi/knobre/okohl/PD/Gripper-HMM/Data/01_Maxfilter/{subject}_gripper_trans.fif"
    )

# Directory to save the preprocessed data to
preproc_dir = "/ohba/pi/knobre/cgohil/pd_gripper/preproc"

# Run batch preprocessing
preprocessing.run_proc_batch(
    config,
    inputs,
    outdir=preproc_dir,
    overwrite=True,
)
