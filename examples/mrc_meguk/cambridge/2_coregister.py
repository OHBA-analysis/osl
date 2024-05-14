"""Coregisteration.

"""
from glob import glob
from pathlib import Path
import numpy as np
from dask.distributed import Client

from osl import source_recon, utils

# Authors : Rukuang Huang <rukuang.huang@jesus.ox.ac.uk>
#           Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

TASK = "resteyesopen"  # resteyesopen or resteyesclosed

BASE_DIR = "/well/woolrich/projects/mrc_meguk/cambridge/ec"
PREPROC_DIR = BASE_DIR + "/preproc"
SRC_DIR = BASE_DIR + "/src"
PREPROC_FILE = PREPROC_DIR + "/{0}_task-{1}_proc-sss_meg/{0}_task-{1}_proc-sss_meg_preproc_raw.fif"
SMRI_FILE = "/well/woolrich/projects/mrc_meguk/raw/Cambridge/{0}/anat/{0}_T1w.nii.gz"


def fix_headshape_points(src_dir, subject, preproc_file, smri_file, epoch_file):
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Drop nasion by 4cm
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

    # if subject == "sub-cam056":
    #     # Remove headshape points that are 1 cm above rpa
    #     keep = hs[2] > (rpa[2] + 10)
    #     hs = hs[:, keep]

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)


config = """
    source_recon:
    - extract_polhemus_from_info: {}
    - fix_headshape_points: {}
    - compute_surfaces:
        include_nose: False
    - coregister:
        use_nose: False
        use_headshape: True
    - forward_model:
        model: Single Layer
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    subjects = []
    smri_files = []
    preproc_files = []

    for directory in sorted(
        glob(PREPROC_DIR + f"/sub*_task-{TASK}_proc-sss_meg")
    ):
        subject = Path(directory).name.split("_")[0]
        subjects.append(subject)
        smri_files.append(SMRI_FILE.format(subject))
        preproc_files.append(PREPROC_FILE.format(subject, TASK))

    client = Client(n_workers=16, threads_per_worker=1)

    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_points],
        dask_client=True,
    )
