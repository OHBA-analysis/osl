"""Source reconstruction: beamforming and parcellation.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np

from osl import source_recon


PREPROC_DIR = "/ohba/pi/knobre/cgohil/covid/preproc"
SRC_DIR = "/ohba/pi/knobre/cgohil/covid/src"
PREPROC_FILE = PREPROC_DIR + "/sub-{0}_task-restEO_preproc_raw.fif"

SUBJECTS = ["004", "005"]

# Settings
config = """
    beamforming:
        freq_range: [1, 45]
        chantypes: meg
        rank: {meg: 60}
    parcellation:
        file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

# Setup
preproc_files = []
for subject in SUBJECTS:
    preproc_files.append(PREPROC_FILE.format(subject))

# Beamforming and parcellation
source_recon.run_src_batch(
    config,
    subjects=SUBJECTS,
    preproc_files=preproc_files,
    src_dir=SRC_DIR,
)
