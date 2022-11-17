"""Example script for preprocessing maxfiltered data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import preprocessing


def crop_ends(dataset, userargs):
    """Crop before the first event and after the last event."""
    raw = dataset["raw"]
    events = dataset["events"]
    tmin = raw.times[events[0][0] - raw.first_samp] - userargs["t"]
    tmax = raw.times[events[-1][0] - raw.first_samp] + userargs["t"]
    if tmax > raw.times[-1]:
        tmax = raw.times[-1]
    raw = raw.crop(tmin=tmin, tmax=tmax)
    dataset["raw"] = raw
    return dataset


config = """
    preproc:
    - find_events: {min_duration: 0.005}
    - crop_ends: {t: 10}
    - notch_filter: {freqs: 50 100 150 200 250}
    - filter: {l_freq: 1, h_freq: 45, fir_design: firwin}
    - resample: {sfreq: 250, npad: auto}
    - ica_raw: {n_components: 0.99, picks: meg}
    - ica_autoreject: {apply: False}
"""

inputs = []
for subject in ["s01_block_01", "s01_block_02"]:
    inputs.append(f"/ohba/pi/knobre/cgohil/dg_int_ext/maxfilter/InEx_{subject}_tsss.fif")

preprocessing.run_proc_batch(
    config,
    inputs,
    outdir="/ohba/pi/knobre/cgohil/dg_int_ext/preproc",
    overwrite=True,
    extra_funcs=[crop_ends],
)
