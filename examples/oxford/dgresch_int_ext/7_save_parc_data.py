"""Save parcellated data as numpy files.

"""

import os
import mne
import numpy as np
from glob import glob

# Directories
event_type = "internal_disp"
src_dir = f"/ohba/pi/knobre/cgohil/dg_int_ext/src/{event_type}"
out_dir = "/ohba/pi/knobre/cgohil/dg_int_ext/src/npy"

os.makedirs(out_dir, exist_ok=True)

# Save epoched data as a numpy file
files = sorted(glob(src_dir + "/*/sflip_parc-epo.fif"))
for i, file in enumerate(files):
    print(f"Saving data: {file} -> {out_dir}/subject{i}.npy")
    epochs = mne.read_epochs(file, verbose=False)
    data = epochs.get_data()  # (epochs, channels, time)
    data = np.swapaxes(data, 1, 2)  # (epochs, time, channels)
    np.save(f"{out_dir}/subject{i}.npy", data)
