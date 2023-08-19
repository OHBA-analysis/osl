"""Example script for epoching parcellated data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import mne

from osl.preprocessing import osl_wrappers

#%% Specify subjects and file paths

# Directories
src_dir = "/ohba/pi/knobre/cgohil/pd_gripper/src"

# Subjects to epoch
subjects = ["HC01", "HC02"]

# Fif files containined parcellated data
parc_files = []
for subject in subjects:
    parc_files.append(f"{src_dir}/{subject}/sflip_parc-raw.fif")

#%% Event info

# IDs for each event type we want to epoch
keys   = ["lowPower_Grip", "highPower_Grip"]
values = [1, 2]
event_id = dict(zip(keys, values))

#%% Epoch

for parc_file in parc_files:
    # Read continuous parcellated data
    raw = mne.io.read_raw_fif(parc_file)

    # Find events
    events = mne.find_events(raw, min_duration=0.005)

    # Epoch
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=-2,
        tmax=5,
        baseline=None,
    )

    # Drop bad segments
    epochs = osl_wrappers.drop_bad_epochs(epochs, picks="misc", metric="var")
    
    # Save
    epoch_file = parc_file.replace("-raw.fif", "-epo.fif")
    print("Saving", epoch_file)
    epochs.save(epoch_file, overwrite=True)
