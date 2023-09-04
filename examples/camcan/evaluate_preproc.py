"""Calculates summary metrics describing the preprocessing.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
from glob import glob

PREPROC_DIR = "/well/woolrich/projects/camcan/winter23/preproc"
LOG_FILES = sorted(
    glob(PREPROC_DIR + f"/logs/mf2pt2_*_ses-rest_task-rest_meg_preproc_raw.log")
)

# Count the number of independent components (ICs) marked as bad
# and the number of bad channels
eogs = {}
ecgs = {}
badchans = {}
for log_file in LOG_FILES:
    with open(log_file, "r") as file:
        lines = file.readlines()
    for line in lines:
        elems = line.split()
        subject = elems[2].split("_")[1]
        if "as EOG ICs" in line:
            eog = int(elems[10])
            eogs[subject] = eog
        if "as ECG ICs" in line:
            ecg = int(elems[10])
            ecgs[subject] = ecg
        if subject not in badchans:
            badchans[subject] = 0
        if "channels rejected" in line:
            badchan = int(elems[12].split("/")[0])
            badchans[subject] += badchan

# Number of subjects with bad ICs and channels
ic_count = 0
badchan_count = 0
for k in eogs.keys():
    if eogs[k] + ecgs[k] > 4:
        ic_count += 1
    if badchans[k] > 0:
        badchan_count += 1

# Average number of ICs
mean_eog = np.mean(list(eogs.values()))
mean_ecg = np.mean(list(ecgs.values()))
ic_mean = mean_eog + mean_ecg

# Average number of bad channels
badchan_mean = np.mean([v for v in badchans.values() if v != 0])

print(f"{ic_count}/{len(eogs)} subjects have more than 4 bad ICs")
print(f"average number of ICs removed is {ic_mean:.2f}")
print(f"{badchan_count}/{len(badchans)} subjects have bad channels")
print(f"average number of bad channels is {badchan_mean:.2f}")
