from pathlib import Path
from glob import glob
from dask.distributed import Client
from osl import source_recon, utils

BASE_DIR = "/well/woolrich/projects/mrc_meguk/cambridge/ec"
PREPROC_DIR = BASE_DIR + "/preproc"
SRC_DIR = BASE_DIR + "/src"
PREPROC_FILE = (
    PREPROC_DIR
    + "/{0}_task-resteyesclosed_proc-sss_meg/{0}_task-resteyesclosed_proc-sss_meg_preproc_raw.fif"
)
SMRI_FILE = "/well/woolrich/projects/mrc_meguk/raw/Cambridge/{0}/anat/{0}_T1w.nii.gz"
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

    subjects = []
    smri_files = []
    preproc_files = []

    for directory in sorted(
        glob(PREPROC_DIR + "/sub*_task-resteyesclosed_proc-sss_meg")
    ):
        subject = Path(directory).name.split("_")[0]
        subjects.append(subject)
        smri_files.append(SMRI_FILE.format(subject))
        preproc_files.append(PREPROC_FILE.format(subject))

    client = Client(n_workers=16, threads_per_worker=1)

    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        dask_client=True,
    )
