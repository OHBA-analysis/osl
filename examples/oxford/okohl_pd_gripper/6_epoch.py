"""Example script for epoching parcellated data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import mne

#%% Specify subjects and file paths

# Directories
src_dir = "/ohba/pi/knobre/cgohil/pd_gripper/src"
epoch_dir = "/ohba/pi/knobre/cgohil/pd_gripper/epoched"

os.makedirs(epoch_dir, exist_ok=True)

# Subjects
subjects = ["HC01", "HC02"]

# Fif files to epoch
parc_files = []
for subject in subjects:
    parc_files.append(f"{src_dir}/{subject}/sflip_parc-raw.fif")

#%% Event info

# IDs for each event type we want to epoch
keys   = ["lowPower_Grip", "highPower_Grip"]
values = [1, 2]
event_id = dict(zip(keys, values))

#%% Epoch

for subject, parc_file in zip(subjects, parc_files):

    # Load Raw object containing the parcellated data
    raw = mne.io.read_raw_fif(parc_file)

    # Find events
    events = mne.find_events(sens_raw, min_duration=0.005)

    # Epoch
    parc_epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=-1,
        tmax=4,
        baseline=None,
    )
    
    # Save
    epoch_file = epoch_dir + f"/{subject}-epo.fif"
    print("Saving", epoch_file)
    parc_epochs.save(epoch_file, overwrite=True)
