"""Source reconstruction: beamforming and parcellation.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np

from osl.source_recon import run_bf_parc_batch

# Directories
PREPROC_DIR = "/ohba/pi/knobre/cgohil/covid/preproc"
SRC_DIR = "/ohba/pi/knobre/cgohil/covid/src"
COREG_DIR = SRC_DIR + "/coreg"
PREPROC_FILE = PREPROC_DIR + "/sub-{0}_task-restEO_preproc_raw.fif"

SUBJECTS = ["004", "005"]

# Files
preproc_files = []
for subject in SUBJECTS:
    preproc_files.append(PREPROC_FILE.format(subject))

# Channels to use
chantypes = ["meg"]
rank = {"meg": 60}

print("Channel types to use:", chantypes)
print("Channel types and ranks for source recon:", rank)

parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

# Beamforming and parcellation
run_bf_parc_batch(
    preproc_files,
    SUBJECTS,
    chantypes,
    rank,
    parcellation_file,
    orthogonalise=True,
    freq_range=[1, 45],
    src_dir=SRC_DIR,
    coreg_dir=COREG_DIR,
)
