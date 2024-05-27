"""Coregistration with RHINO.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os

from osl import source_recon, utils

def coregister(src_dir, subject, preproc_file, smri_file, epoch_file):
    """Coregister OPM data."""

    # Create dummy coregistration files
    source_recon.rhino.coreg(
        fif_file=preproc_file,
        subjects_dir=src_dir,
        subject=subject,
        use_headshape=False,
        use_nose=False,
        already_coregistered=True,
    )

    # Copy head to MRI transformation (needed for the forward model)
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)
    in_file = f"data/raw/{subject}-head_scaledmri-trans.fif"    
    out_file = filenames["head_scaledmri_t_file"]
    cmd = f"cp {in_file} {out_file}"
    utils.logger.log_or_print(cmd)
    os.system(cmd)

# Settings
config = """
    source_recon:
    - compute_surfaces:
        include_nose: False
    - coregister: {}
    - forward_model:
        model: Single Layer
"""

# Subject IDs
subjects = ["13703"]

# Fif files containing the sensor-level preprocessed data for each subject
preproc_files = [
    "data/preproc/13703-braille_test-meg/13703-braille_test-meg_preproc_raw.fif",
]

# The corresponding structurals for each subject
smri_files = ["data/raw/13703.nii"]

# Directory to save output to
outdir = "data/coreg"

# Perform coregistration
source_recon.run_src_batch(
    config,
    src_dir=outdir,
    subjects=subjects,
    preproc_files=preproc_files,
    smri_files=smri_files,
    extra_funcs=[coregister],
)
