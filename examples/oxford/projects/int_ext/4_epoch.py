"""Example script for epoching preprocessed data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import preprocessing

# Setup paths to preprocessed data files
preproc_dir = "/ohba/pi/knobre/cgohil/int_ext/preproc"
subjects = ["s01_block_01", "s01_block_02"]

run_ids = []
preproc_files = []
for subject in subjects:
    run_id = f"InEx_{subject}_tsss"
    run_ids.append(run_id)
    preproc_files.append(f"{preproc_dir}/{run_id}/{run_id}_preproc_raw.fif")

# Setup IDs for each event type we want to epoch
# In this script we will epoch trials related to a particular event type
event_type = "internal_disp"

if event_type == "external_disp":
    keys   = ["int2int_TL", "int2int_TR", "int2int_BL", "int2int_BR",
              "ext2int_TL", "ext2int_TR", "ext2int_BL", "ext2int_BR",
              "int2none_TL", "int2none_TR", "int2none_BL", "int2none_BR",
              "ext2ext_TL", "ext2ext_TR", "ext2ext_BL", "ext2ext_BR",
              "int2ext_TL", "int2ext_TR", "int2ext_BL", "int2ext_BR",
              "ext2none_TL", "ext2none_TR", "ext2none_BL", "ext2none_BR"]
    values = [101,102,103,104,
              121,122,123,124,
              141,142,143,144,
              111,112,113,114,
              131,132,133,134,
              151,152,153,154]

elif event_type == "first_cue":
    keys   = ["1stCue_internal_within_TL", "1stCue_internal_within_TR", "1stCue_internal_within_BL", "1stCue_internal_within_BR",
              "1stCue_internal_between_TL", "1stCue_internal_between_TR", "1stCue_internal_between_BL", "1stCue_internal_between_BR",
              "1stCue_internal_none_TL", "1stCue_internal_none_TR", "1stCue_internal_none_BL", "1stCue_internal_none_BR",
              "1stCue_external_within_TL", "1stCue_external_within_TR", "1stCue_external_within_BL", "1stCue_external_within_BR",
              "1stCue_external_between_TL", "1stCue_external_between_TR", "1stCue_external_between_BL", "1stCue_external_between_BR",
              "1stCue_external_none_TL", "1stCue_external_none_TR", "1stCue_external_none_BL", "1stCue_external_none_BR"]
    values = [1, 2, 3, 4, 21, 22, 23, 24, 41, 42, 43, 44, 11, 12, 13, 14, 31, 32, 33, 34, 51, 52, 53, 54]

elif event_type == "internal_disp":
    keys   = ["TLTR", "TLBR", "BLTR", "BLBR"]
    values = [171,172,173,174]

event_codes = dict(zip(keys, values))

# Settings for epoching
config = f"""
    meta:
      event_codes: {event_codes}
    preproc:
    - find_events:
        min_duration: 0.005
    - crop_ends:
        t: 10
    - epochs:
        tmin: -0.25
        tmax: 1
        baseline: null
    - drop_bad_epochs:
        picks: mag
        metric: var
    - drop_bad_epochs:
        picks: grad
        metric: var
"""

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

# Output directory
epoch_dir = f"/ohba/pi/knobre/cgohil/int_ext/epoch/{event_type}"

# Run epoching
preprocessing.run_proc_batch(
    config,
    preproc_files,
    outdir=epoch_dir,
    overwrite=True,
    extra_funcs=[crop_ends],
)
