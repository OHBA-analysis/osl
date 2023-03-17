"""Save epoched parcellated data as a numpy file.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import mne
import numpy as np

#%% Specify subjects and file paths

# Directories
epochs_dir = "/ohba/pi/knobre/cgohil/pd_gripper/epoched"

os.makedirs(epochs_dir + "/npy", exist_ok=True)

# Subjects
subjects = ["HC01", "HC02"]

#%% Save epoched data as numpy files

for i, subject in enumerate(subjects):
    epochs_file = f"{epochs_dir}/{subject}-epo.fif"
    print(f"Saving data: {epochs_file} -> {epochs_dir}/npy/subject{i}.npy")
    epochs = mne.read_epochs(epochs_file, verbose=False)
    data = epochs.get_data()  # (epochs, channels, time)
    data = np.swapaxes(data, 1, 2)  # (epochs, time channels)
    np.save(f"{epochs_dir}/npy/subject{i}.npy", data)
