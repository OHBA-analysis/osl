"""Coregistration.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np

from osl import source_recon, utils


config = """
    source_recon:   
    - extract_polhemus_from_info: {}
    - save_mni_fiducials:
        filepath: data/fiducials/{subject}_smri_fid.txt
    - compute_surfaces:
        include_nose: False    
    - coregister:
        use_nose: False
        use_headshape: False
"""

# List of subject IDs
subjects = ["LN_VTA2"]

# Lists for input files
preproc_files = ["data/preproc/mg04938_BrainampDBS_20170504_01_preproc_raw.fif"]
smri_files = ["data/smri/LN_VTA2.nii"]

# Output directory
coreg_dir = "data/coreg"

# Do coregistration
source_recon.run_src_batch(
    config,
    src_dir=coreg_dir,
    subjects=subjects,
    preproc_files=preproc_files,
    smri_files=smri_files,
)
