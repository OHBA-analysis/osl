"""Source reconstruction using a LCMV beamformer and parcellation.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import source_recon

# Settings
config = """
    source_recon:
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: mag
        rank: {mag: 120}
        parcellation_file: aal_cortical_merged_8mm_stacked.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

# Subject IDs
subjects = [
    "sub-not001_task-resteyesopen",
    "sub-not002_task-resteyesopen",
]

# Fif files containing the sensor-level preprocessed data for each subject
preproc_files = [
    "data/sub-not001_task-resteyesopen/sub-not001_task-resteyesopen_preproc_raw.fif",
    "data/sub-not002_task-resteyesopen/sub-not002_task-resteyesopen_preproc_raw.fif",
]

# Directory to save output to
outdir = "data"

# Source reconstruction
source_recon.run_src_batch(
    config,
    outdir=outdir,
    subjects=subjects,
    preproc_files=preproc_files,
)
