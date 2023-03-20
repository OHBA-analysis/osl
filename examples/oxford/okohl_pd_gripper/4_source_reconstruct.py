"""Source reconstruct continuous sensor-level data.

"""

# Authors: Oliver Kohl <oliver.kohl@psych.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np

from osl import source_recon

#%% Setup FSL

# This is the directory on hbaws that contains FSL
source_recon.setup_fsl("/opt/ohba/fsl/6.0.5")

#%% Specify subjects and file paths

# Subjects
subjects = ["HC01", "HC02"]

# Directories
preproc_dir = "/ohba/pi/knobre/cgohil/pd_gripper/preproc"
src_dir = "/ohba/pi/knobre/cgohil/pd_gripper/src"

# Setup paths
preproc_files = []
for subject in subjects:
    preproc_files.append(f"{preproc_dir}/{subject}_gripper_trans/{subject}_gripper_trans_preproc_raw.fif")

#%% Source reconstruction (forward modelling, beamforming and parcellation)

# Settings
config = """
    source_recon:
    - forward_model:
        model: Single Layer
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: [mag, grad]
        rank: {meg: 60}
        parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

# Run source reconstruction
source_recon.run_src_batch(
    config,
    src_dir=src_dir,
    subjects=subjects,
    preproc_files=preproc_files,
)
