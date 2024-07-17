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
preproc_files = ["data/LN_VTA2/mg04938_BrainampDBS_20170504_01_preproc_raw.fif"]
smri_files = ["smri/LN_VTA2.nii"]

# Output directory
outdir = "data"

# Do coregistration
source_recon.run_src_batch(
    config,
    outdir=outdir,
    subjects=subjects,
    preproc_files=preproc_files,
    smri_files=smri_files,
)
