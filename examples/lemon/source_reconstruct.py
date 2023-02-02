"""Example script for source reconstructing the LEMON dataset.

This script uses an external parcellation file (outside of the osl package)
and processes the data files in parallel.
"""

# Authors : Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
import pathlib
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

import logging
logger = logging.getLogger("osl")

# Directories
BASE_DIR = "/well/woolrich/projects/lemon"
RAW_DIR = BASE_DIR + "/raw"
PREPROC_DIR = BASE_DIR + "/osl_example/preproc"
SRC_DIR = BASE_DIR + "/osl_example/src"
FSL_DIR = "/well/woolrich/projects/software/fsl"

# Files
PREPROC_FILE = PREPROC_DIR + "/{0}/{0}_preproc_raw.fif"
SMRI_FILE = RAW_DIR + "/{0}/ses-01/anat/{0}_ses-01_inv-2_mp2rage.nii.gz"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(FSL_DIR)

    # Settings
    config = """
        source_recon:
        - extract_fiducials_from_fif:
            include_eeg_as_headshape: true
        - fix_headshape_fiducials: {}
        - compute_surfaces_coregister_and_forward_model:
            include_nose: false
            use_nose: false
            use_headshape: false
            model: Triple Layer
            eeg: true
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: eeg
            rank: {eeg: 50}
            parcellation_file: parcellations/Yeo17_8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
    """

    def fix_headshape_fiducials(src_dir, subject, preproc_file, smri_file, epoch_file):
        # Get coreg filenames
        filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

        # Load saved headshape and nasion files
        hs = np.loadtxt(filenames["polhemus_headshape_file"])
        nas = np.loadtxt(filenames["polhemus_nasion_file"])
        lpa = np.loadtxt(filenames["polhemus_lpa_file"])
        rpa = np.loadtxt(filenames["polhemus_rpa_file"])

        # Shrink headshape points by 5%
        hs *= 0.95

        # Move fiducials down 1 cm
        nas[2] -= 10
        lpa[2] -= 10
        rpa[2] -= 10

        # Move fiducials back 1 cm
        nas[1] -= 10
        lpa[1] -= 10
        rpa[1] -= 10

        # Overwrite files
        logger.info(f"overwritting {filenames['polhemus_nasion_file']}")
        np.savetxt(filenames["polhemus_nasion_file"], nas)
        logger.info(f"overwritting {filenames['polhemus_lpa_file']}")
        np.savetxt(filenames["polhemus_lpa_file"], lpa)
        logger.info(f"overwritting {filenames['polhemus_rpa_file']}")
        np.savetxt(filenames["polhemus_rpa_file"], rpa)
        logger.info(f"overwritting {filenames['polhemus_headshape_file']}")
        np.savetxt(filenames["polhemus_headshape_file"], hs)

    # Get subjects
    subjects = []
    for subject in sorted(glob(PREPROC_FILE.replace("{0}", "*"))):
        subjects.append(pathlib.Path(subject).stem.split("_")[0])

    # Get files
    preproc_files = []
    smri_files = []
    for subject in subjects:
        preproc_files.append(PREPROC_FILE.format(subject))
        smri_files.append(SMRI_FILE.format(subject))

    # Setup parallel processing
    client = Client(n_workers=2, threads_per_worker=1)

    # Coregistration
    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_fiducials],
        dask_client=True,
    )
