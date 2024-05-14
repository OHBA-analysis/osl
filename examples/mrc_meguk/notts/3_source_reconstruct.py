"""Source reconstruction.

"""

import os.path as op
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import Client

from osl import source_recon, utils

# Authors : Rukuang Huang <rukuang.huang@jesus.ox.ac.uk>
#           Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

TASK = "resteyesopen"  # resteyesopen or resteyesclosed

# Directories
RAW_DIR = "/well/woolrich/projects/mrc_meguk/raw/Nottingham"
PREPROC_DIR = f"/well/woolrich/projects/mrc_meguk/notts/{TASK}/preproc"
SRC_DIR = f"/well/woolrich/projects/mrc_meguk/notts/{TASK}/src"

SMRI_FILE = f"/well/woolrich/projects/mrc_meguk/notts/{TASK}/smri/{0}_T1w.nii.gz"
PREPROC_FILE = PREPROC_DIR + "/{0}_task-resteyesopen_meg/{0}_task-" + TASK + "_meg_preproc_raw.fif"
POS_FILE = RAW_DIR + "/{0}/meg/{0}_headshape.pos"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    # Settings
    config = f"""
        source_recon:
        - save_polhemus_from_pos:
            pos_filepath: {POS_FILE}
        - compute_surfaces:
            include_nose: True
        - coregister:
            use_nose: True
            use_headshape: True
        - forward_model:
            model: Single Layer
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: mag
            rank: {mag: 120}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
    """

    # Get input files
    subjects = []
    smri_files = []
    preproc_files = []
    for path in sorted(glob(PREPROC_FILE.replace("{0}", "*"))):
        subject = Path(path).stem.split("_")[0]
        subjects.append(subject)
        preproc_files.append(PREPROC_FILE.format(subject))
        smri_files.append(SMRI_FILE.format(subject))

    # Source reconstruction
    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        dask_client=True,
    )
