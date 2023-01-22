"""Source reconstruction: forward modelling, beamforming and parcellation.

Note, before this script is run the /coreg directory created by coregister.py
must be copied and renamed to /src.
"""

import pathlib
from glob import glob
from dask.distributed import Client
from osl import source_recon, utils

BASE_DIR = "/well/woolrich/projects/camcan"
PREPROC_DIR = BASE_DIR + "/winter23/preproc"
SRC_DIR = BASE_DIR + "/winter23/src"
ANAT_DIR = BASE_DIR + "/cc700/mri/pipeline/release004/BIDS_20190411/anat"
PREPROC_FILE = PREPROC_DIR + "/mf2pt2_{0}_ses-rest_task-rest_meg/mf2pt2_{0}_ses-rest_task-rest_meg_preproc_raw.fif"
SMRI_FILE = ANAT_DIR + "/{0}/anat/{0}_T1w.nii"
FSL_DIR = "/well/woolrich/projects/software/fsl"

config = """
    source_recon:
    - forward_model:
        model: Single Layer
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: [mag, grad]
        rank: {meg: 60}
        parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(FSL_DIR)

    # Get subjects
    subjects = []
    for subject in sorted(
        glob(
            PREPROC_DIR
            + "/mf2pt2_*_ses-rest_task-rest_meg"
            + "/mf2pt2_sub-*_ses-rest_task-rest_meg_preproc_raw.fif"
        )
    ):
        subjects.append(pathlib.Path(subject).stem.split("_")[1])

    # Setup files
    smri_files = []
    preproc_files = []
    for subject in subjects:
        smri_files.append(SMRI_FILE.format(subject))
        preproc_files.append(PREPROC_FILE.format(subject))

    # Setup parallel processing
    client = Client(n_workers=16, threads_per_worker=1)

    # Run beamforming and parcellation
    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        dask_client=True,
    )
