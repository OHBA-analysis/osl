"""Source reconstruction: forward modelling, beamforming and parcellation.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import pathlib
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

# Directories
BASE_DIR = "/well/woolrich/projects/camcan"
PREPROC_DIR = BASE_DIR + "/preproc"
COREG_DIR = BASE_DIR + "/coreg"
SRC_DIR = BASE_DIR + "/src"

# Files
PREPROC_FILE = (
    PREPROC_DIR
    + "/mf2pt2_{0}_ses-rest_task-rest_meg"
    + "/mf2pt2_{0}_ses-rest_task-rest_meg_preproc_raw.fif"
)

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Copy coreg directory
    if not os.path.exists(SRC_DIR):
        cmd = f"cp -r {COREG_DIR} {SRC_DIR}"
        print(cmd)
        os.system(cmd)

    # Get subjects
    subjects = []
    for subject in sorted(glob(PREPROC_FILE.replace("{0}", "*"))):
        subjects.append(pathlib.Path(subject).stem.split("_")[1])

    # Setup files
    preproc_files = []
    for subject in subjects:
        preproc_files.append(PREPROC_FILE.format(subject))

    # Settings
    config = """
        source_recon:
        - forward_model:
            model: Single Layer
        - beamform_and_parcellate:
            freq_range: [1, 80]
            chantypes: [mag, grad]
            rank: {meg: 60}
            parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
    """

    # Setup parallel processing
    client = Client(n_workers=16, threads_per_worker=1)

    # Run beamforming and parcellation
    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        dask_client=True,
    )
