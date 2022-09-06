"""Source reconstruction: beamforming, parcellation and orthogonalisation.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
import pathlib
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils


PREPROC_DIR = "/ohba/pi/mwoolrich/cgohil/camcan/preproc"
SRC_DIR = "/ohba/pi/mwoolrich/cgohil/camcan/src"
FSL_DIR = "/home/cgohil/local/fsl"
PREPROC_FILE = PREPROC_DIR + "/{0}_ses-rest_task-rest_meg_preproc_raw.fif"

# Settings
config = """
    beamforming:
        freq_range: [1, 45]
        chantypes: meg
        ranks: 60
    parcellation:
        file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

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

    # Setup parallel processing
    client = Client(n_workers=2, threads_per_worker=1)

    # Beamforming and parcellation
    source_recon.run_src_batch(
        config,
        subjects=subjects,
        preproc_files=preproc_files,
        src_dir=SRC_DIR,
        dask_client=True,
    )
