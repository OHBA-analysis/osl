"""Source reconstruction: beamforming, parcellation and orthogonalisation.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
import pathlib
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

# Directories
PREPROC_DIR = "/ohba/pi/mwoolrich/cgohil/camcan/preproc"
SRC_DIR = "/ohba/pi/mwoolrich/cgohil/camcan/src"
COREG_DIR = SRC_DIR + "/coreg"
FSL_DIR = "/home/cgohil/local/fsl"

# Files
PREPROC_FILE = PREPROC_DIR + "/{0}_ses-rest_task-rest_meg_preproc_raw.fif"
PARCELLATION_FILE = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup
    source_recon.setup_fsl(FSL_DIR)

    # Get subjects
    subjects = []
    for subject in glob(PREPROC_DIR + "/sub-*"):
        subjects.append(pathlib.Path(subject).stem.split("_")[0])

    # Files
    preproc_files = []
    for subject in subjects:
        preproc_files.append(PREPROC_FILE.format(subject))

    # Channels to use
    chantypes = ["meg"]
    rank = {"meg": 60}

    print("Channel types to use:", chantypes)
    print("Channel types and ranks for source recon:", rank)

    # Setup parallel processing
    client = Client(n_workers=2, threads_per_worker=1)

    # Beamforming and parcellation
    source_recon.run_bf_parc_batch(
        preproc_files,
        subjects,
        chantypes,
        rank,
        PARCELLATION_FILE,
        orthogonalise=True,
        freq_range=[1, 45],
        src_dir=SRC_DIR,
        coreg_dir=COREG_DIR,
        dask_client=True,
    )
