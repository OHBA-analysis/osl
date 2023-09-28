"""Example script for preprocessing maxfiltered data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import preprocessing

# Settings for preprocessing
config = """
    preproc:
    - notch_filter: {freqs: 50 100 150 200 250}
    - filter: {l_freq: 1, h_freq: 45, fir_design: firwin}
    - resample: {sfreq: 250, npad: auto}
    - ica_raw: {n_components: 0.99, picks: meg}
    - ica_autoreject: {apply: False}
"""

# Setup paths to maxfiltered data files
inputs = []
for subject in ["s01_block_01", "s01_block_02"]:
    inputs.append(f"/ohba/pi/knobre/cgohil/int_ext/maxfilter/InEx_{subject}_tsss.fif")

# Directory to save the preprocessed data to
preproc_dir = "/ohba/pi/knobre/cgohil/int_ext/preproc"

# Run batch preprocessing
preprocessing.run_proc_batch(
    config,
    inputs,
    outdir=preproc_dir,
    overwrite=True,
)
