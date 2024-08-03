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
anatdir = "/ohba/pi/mwoolrich/datasets/CamCan_2021/cc700/mri/pipeline/release004/BIDS_20190411/anat"
outdir = "/ohba/pi/mwoolrich/cgohil/camcan/src"

# Files
smri_file = anatdir + "/{0}/anat/{0}_T1w.nii"
preproc_file = outdir + "{0}_ses-rest_task-rest_meg/{0}_ses-rest_task-rest_meg_preproc-raw.fif"

# Settings
config = """
    source_recon:
    - extract_polhemus_from_info: {}
    - remove_headshape_points: {}
    - compute_surfaces:
        include_nose: False
    - coregister:
        use_nose: False
        use_headshape: True
    - forward_model:
        model: Single Layer
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: meg
        rank: {meg: 60}
        parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

def remove_headshape_points(outdir, subject):
    """Removes headshape points near the nose."""

    # Get coreg filenames
    filenames = source_recon.rhino.get_coreg_filenames(outdir, subject)

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

# Get subjects
subjects = []
for subject in sorted(glob(f"{outdir}/sub-*")):
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
    outdir=outdir,
    subjects=subjects,
    preproc_files=preproc_files,
    smri_files=smri_files,
    extra_funcs=[remove_headshape_points],
)
