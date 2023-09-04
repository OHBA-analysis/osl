"""Source reconstruction: forward modelling, beamforming and parcellation.

Note, before this script is run the /coreg directory created by coregister.py
must be copied and renamed to /src.
"""

import pathlib
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

# Directories
BASE_DIR = "/well/woolrich/projects/camcan"
PREPROC_DIR = BASE_DIR + "/summer23/preproc"
SRC_DIR = BASE_DIR + "/summer23/src"
FSL_DIR = "/well/woolrich/projects/software/fsl"

# Files
PREPROC_FILE = (
    PREPROC_DIR
    + "/mf2pt2_{0}_ses-rest_task-rest_meg"
    + "/mf2pt2_{0}_ses-rest_task-rest_meg_preproc_raw.fif"
)

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
        extra_chans: [eog, ecg]
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(FSL_DIR)

    # Get subjects
    subjects = []
    for subject in sorted(glob(PREPROC_FILE.replace("{0}", "*"))):
        subjects.append(pathlib.Path(subject).stem.split("_")[1])

    # Setup files
    preproc_files = []
    for subject in subjects:
        preproc_files.append(PREPROC_FILE.format(subject))

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
