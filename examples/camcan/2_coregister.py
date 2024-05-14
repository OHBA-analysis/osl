"""Coregisteration.

The scripts was first run for all subjects (with n_init=1). Then for subjects
whose coregistration looked a bit off we re-run this script just for that
particular subject with a higher n_init.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
import pathlib
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

# Directories
BASE_DIR = "/well/woolrich/projects/camcan"
PREPROC_DIR = BASE_DIR + "/summer23/preproc"
COREG_DIR = BASE_DIR + "/summer23/coreg"
ANAT_DIR = BASE_DIR + "/cc700/mri/pipeline/release004/BIDS_20190411/anat"

# Files
PREPROC_FILE = (
    PREPROC_DIR
    + "/mf2pt2_{0}_ses-rest_task-rest_meg"
    + "/mf2pt2_{0}_ses-rest_task-rest_meg_preproc_raw.fif"
)
SMRI_FILE = ANAT_DIR + "/{0}/anat/{0}_T1w.nii"

def fix_headshape_points(src_dir, subject, preproc_file, smri_file, epoch_file):
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Drop nasion by 4cm and remove headshape points more than 7 cm away
    nas[2] -= 40
    distances = np.sqrt(
        (nas[0] - hs[0]) ** 2 + (nas[1] - hs[1]) ** 2 + (nas[2] - hs[2]) ** 2
    )
    keep = distances > 70
    hs = hs[:, keep]

    # Remove anything outside of rpa
    keep = hs[0] < rpa[0]
    hs = hs[:, keep]

    # Remove anything outside of lpa
    keep = hs[0] > lpa[0]
    hs = hs[:, keep]

    # Remove headshape points on the neck
    remove = hs[2] < min(lpa[2], rpa[2]) - 4
    hs = hs[:, ~remove]

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)


if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Get subjects
    subjects = []
    for subject in sorted(glob(PREPROC_FILE.replace("{0}", "*"))):
        subjects.append(pathlib.Path(subject).stem.split("_")[1])

    # Setup files
    smri_files = []
    preproc_files = []
    for subject in subjects:
        smri_files.append(SMRI_FILE.format(subject))
        preproc_files.append(PREPROC_FILE.format(subject))

    # Settings
    config = """
        source_recon:
        - extract_polhemus_from_info: {}
        - fix_headshape_points: {}
        - compute_surfaces:
            include_nose: False
        - coregister:
            use_nose: False
            use_headshape: True
            #n_init: 50
    """

    # Setup parallel processing
    client = Client(n_workers=16, threads_per_worker=1)

    # Run coregistration
    source_recon.run_src_batch(
        config,
        src_dir=COREG_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_points],
        dask_client=True,
    )
