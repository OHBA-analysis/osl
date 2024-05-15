"""Source reconstruct preprocessed uk_meg_notts data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from glob import glob
from pathlib import Path
from dask.distributed import Client

from osl import source_recon, utils


BASE_DIR = "/well/woolrich/projects/uk_meg_notts/eo/oslpy22"
RAW_DIR = BASE_DIR + "/raw"
PREPROC_DIR = BASE_DIR + "/preproc"
SRC_DIR = BASE_DIR + "/src"

SMRI_FILE = RAW_DIR + "/{0}/{0}.nii"
PREPROC_FILE = PREPROC_DIR + "/{0}_raw/{0}_preproc_raw.fif"
POS_FILE = RAW_DIR + "/{0}/{0}.pos"

config = f"""
    source_recon:
    - extract_polhemus_from_pos:
        filepath: {POS_FILE}
    - compute_surfaces:
        include_nose: True
    - coregister:
        use_nose: True
        use_headshape: Rrue
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

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    subjects = []
    preproc_files = []
    smri_files = []
    for path in sorted(glob(PREPROC_DIR + "/*/*_preproc_raw.fif")):
        subject = Path(path).stem.split("_")[0]
        subjects.append(subject)
        preproc_files.append(PREPROC_FILE.format(subject))
        smri_files.append(SMRI_FILE.format(subject))

    client = Client(n_workers=16, threads_per_worker=1)

    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        dask_client=True,
    )
