"""Save parcellated data as numpy files.

"""

import os
import mne
import numpy as np
from glob import glob

# Directories
src_dir = "/ohba/pi/knobre/cgohil/covid/src"
out_dir = f"{src_dir}/npy"

os.makedirs(out_dir, exist_ok=True)

# Save parcellated data as a numpy file
files = sorted(glob(src_dir + "/*/sflip_parc-raw.fif"))
for i, file in enumerate(files):
    print(f"Saving data: {file} -> {out_dir}/subject{i}.npy")
    raw = mne.io.read_raw_fif(file, verbose=False)
    data = raw.get_data(reject_by_annotation="omit").T  # (time, channels)
    np.save(f"{out_dir}/subject{i}.npy", data)
