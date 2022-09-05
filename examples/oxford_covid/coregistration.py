"""Coregistration for source reconstruction.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
from osl import rhino

# Directories
RAW_DIR = "/ohba/pi/knobre/datasets/covid/rawbids"
PREPROC_DIR = "/ohba/pi/knobre/cgohil/covid/preproc"
SRC_DIR = "/ohba/pi/knobre/cgohil/covid/src"
COREG_DIR = SRC_DIR + "/coreg"

# Files
RAW_FILE = RAW_DIR + "/sub-{0}/meg/sub-{0}_task-restEO.fif"
SMRI_FILE = RAW_DIR + "/sub-{0}/anat/sub-{0}_T1w.nii"
PREPROC_FILE = PREPROC_DIR + "/sub-{0}_task-restEO_preproc_raw.fif"

SUBJECTS = ["004", "005"]

# Setup
rhino.utils.setup_fsl("/home/cgohil/local/fsl")

raw_files = []
smri_files = []
preproc_files = []
for subject in SUBJECTS:
    raw_files.append(RAW_FILE.format(subject))
    smri_files.append(SMRI_FILE.format(subject))
    preproc_files.append(PREPROC_FILE.format(subject))

# Coregistration
rhino.run_coreg_batch(
    coreg_dir=COREG_DIR,
    subjects=SUBJECTS,
    raw_files=raw_files,
    preproc_files=preproc_files,
    smri_files=smri_files,
    model="Single Layer",
    include_nose=True,
    use_nose=True,
)
