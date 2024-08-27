"""Source reconstruction using a LCMV beamformer and parcellation.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import source_recon

# Settings
config = """
    source_recon:
    - forward_model:
        model: Single Layer
    - beamform_and_parcellate:
        freq_range: [1, 80]
        chantypes: mag
        rank: {mag: 120}
        parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

# List of subject IDs
subjects = ["LN_VTA2"]

# Fif files containing the sensor-level preprocessed data for each subject
preproc_files = ["data/LN_VTA2/mg04938_BrainampDBS_20170504_01_preproc-raw.fif"]

# Directories
outdir = "data"

# Source reconstruction
source_recon.run_src_batch(
    config,
    outdir=outdir,
    subjects=subjects,
    preproc_files=preproc_files,
)
