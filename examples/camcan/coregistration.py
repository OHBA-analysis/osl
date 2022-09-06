"""Coregister CamCAN for source reconstruction.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
import pathlib
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

# Directories
ANAT_DIR = "/ohba/pi/mwoolrich/datasets/CamCan_2021/cc700/mri/pipeline/release004/BIDS_20190411/anat"
PREPROC_DIR = "/ohba/pi/mwoolrich/cgohil/camcan/preproc"
SRC_DIR = "/ohba/pi/mwoolrich/cgohil/camcan/src"
COREG_DIR = SRC_DIR + "/coreg"

# Files
SMRI_FILE = ANAT_DIR + "/{0}/anat/{0}_T1w.nii"
PREPROC_FILE = PREPROC_DIR + "/{0}_ses-rest_task-rest_meg_preproc_raw.fif"


def remove_points(
    polhemus_headshape_file,
    polhemus_nasion_file,
    **kwargs,
):
    """Removes headshape points near the nose."""
    hs = np.loadtxt(polhemus_headshape_file)
    nas = np.loadtxt(polhemus_nasion_file)
    nas[2] -= 40  # drop nasion by 4cm
    distances = np.sqrt(
        (nas[0] - hs[0]) ** 2 + (nas[1] - hs[1]) ** 2 + (nas[2] - hs[2]) ** 2
    )
    keep = distances > 70  # keep headshape points more than 7cm away
    hs = hs[:, keep]
    np.savetxt(polhemus_headshape_file, hs)


if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Get subjects
    subjects = []
    for subject in glob(PREPROC_DIR + "/sub-*"):
        subjects.append(pathlib.Path(subject).stem.split("_")[0])

    # Setup
    source_recon.setup_fsl("/home/cgohil/local/fsl")

    smri_files = []
    preproc_files = []
    for subject in subjects:
        smri_files.append(SMRI_FILE.format(subject))
        preproc_files.append(PREPROC_FILE.format(subject))

    # Setup parallel processing
    client = Client(n_workers=2, threads_per_worker=1)

    # Coregistration
    source_recon.run_coreg_batch(
        coreg_dir=COREG_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        model="Single Layer",
        include_nose=False,
        use_nose=False,
        edit_polhemus_func=remove_points,
        dask_client=True,
    )
