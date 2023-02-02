"""Example script for source reconstructing CamCAN in serial.

In this script, we source reconstruct each subject one at a time.

Source reconstruction include coregistration, beamforming, parcellation and
orthogonalisation.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
import pathlib
import os.path as op
from glob import glob

from osl import source_recon

import logging
logger = logging.getLogger("osl")

# Directories
anat_dir = "/ohba/pi/mwoolrich/datasets/CamCan_2021/cc700/mri/pipeline/release004/BIDS_20190411/anat"
preproc_dir = "/ohba/pi/mwoolrich/cgohil/camcan/preproc"
src_dir = "/ohba/pi/mwoolrich/cgohil/camcan/src"
fsl_dir = "/home/cgohil/local/fsl"

# Files
smri_file = anat_dir + "/{0}/anat/{0}_T1w.nii"
preproc_file = preproc_dir + "{0}_ses-rest_task-rest_meg/{0}_ses-rest_task-rest_meg_preproc_raw.fif"

# Settings
config = """
    source_recon:
    - extract_fiducials_from_fif: {}
    - remove_headshape_points: {}
    - compute_surfaces_coregister_and_forward_model:
        include_nose: false
        use_nose: false
        use_headshape: true
        model: Single Layer
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: meg
        rank: {meg: 60}
        parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

def remove_headshape_points(src_dir, subject, preproc_file, smri_file, epoch_file):
    """Removes headshape points near the nose."""

    # Get coreg filenames
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])

    # Drop nasion by 4cm
    nas[2] -= 40  
    distances = np.sqrt(
        (nas[0] - hs[0]) ** 2 + (nas[1] - hs[1]) ** 2 + (nas[2] - hs[2]) ** 2
    )

    # Keep headshape points more than 7cm away
    keep = distances > 70  
    hs = hs[:, keep]

    # Overwrite headshape file
    logger.info(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)

# Setup FSL
source_recon.setup_fsl(fsl_dir)

# Get subjects
subjects = []
for subject in glob(preproc_dir + "/sub-*"):
    subjects.append(pathlib.Path(subject).stem.split("_")[0])

# Setup files
smri_files = []
preproc_files = []
for subject in subjects:
    smri_files.append(smri_file.format(subject))
    preproc_files.append(preproc_file.format(subject))

# Beamforming and parcellation
source_recon.run_src_batch(
    config,
    src_dir=src_dir,
    subjects=subjects,
    preproc_files=preproc_files,
    smri_files=smri_files,
    extra_funcs=[remove_headshape_points],
)
