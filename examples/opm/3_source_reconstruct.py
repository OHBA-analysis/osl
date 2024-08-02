"""Source reconstruction with an LCMV beamformer.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os

from osl import source_recon

# Directories
outdir = "data"

# Settings
config = """
    source_recon:
    - beamform_and_parcellate:
        freq_range: [4, 40]
        chantypes: mag
        rank: {mag: 100}
        spatial_resolution: 8
        parcellation_file: aal_cortical_merged_8mm_stacked.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

# Subject IDs
subjects = ["13703"]

# Fif files containing the sensor-level preprocessed data for each subject
preproc_files = [
    "data/13703/13703-braille_test-meg_preproc-raw.fif",
]

# Do source reconstruction
source_recon.run_src_batch(
    config,
    outdir=outdir,
    subjects=subjects,
    preproc_files=preproc_files,
)
