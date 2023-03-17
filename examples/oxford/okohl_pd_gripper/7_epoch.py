"""Example script for epoching parcelalted data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import mne

#%% Specify subjects and file paths

# Directories
preproc_dir = "/ohba/pi/knobre/cgohil/pd_gripper/preproc"
src_dir = "/ohba/pi/knobre/cgohil/pd_gripper/src"
epoch_dir = "/ohba/pi/knobre/cgohil/pd_gripper/epoched"

os.makedirs(epoch_dir, exist_ok=True)

# Subjects
subjects = ["HC01", "HC02"]

# Fif files to epoch
sens_files = []
parc_files = []
for subject in subjects:
    sens_files.append(
        f"{preproc_dir}/{subject}_gripper_trans/{subject}_gripper_trans_preproc_raw.fif"
    )
    parc_files.append(f"{src_dir}/{subject}/rhino/sflip_parc-raw.fif")

#%% Event info

# IDs for each event type we want to epoch
keys   = ["lowPower_Grip", "highPower_Grip"]
values = [1, 2]
event_id = dict(zip(keys, values))

#%% Epoch

for subject, sens_file, parc_file in zip(subjects, sens_files, parc_files):
    # Load Raw object containing preprocessed sensor data
    sens_raw = mne.io.read_raw_fif(sens_file)

    # Find events
    events = mne.find_events(sens_raw, min_duration=0.005)

    # Load Raw object containing the parcellated data
    parc_raw = mne.io.read_raw_fif(parc_file)

    # Epoch
    parc_epochs = mne.Epochs(
        parc_raw,
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
