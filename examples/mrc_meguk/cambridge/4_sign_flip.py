"""Dipole sign flipping.

"""

from glob import glob
from pathlib import Path
from dask.distributed import Client
from osl import utils

from osl.source_recon import find_template_subject, run_src_batch, setup_fsl

# Authors : Rukuang Huang <rukuang.huang@jesus.ox.ac.uk>
#           Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

TASK = "resteyesopen"  # resteyesopen or resteyesclosed

BASE_DIR = f"/well/woolrich/projects/mrc_meguk/cambridge/{TASK}"
PREPROC_DIR = BASE_DIR + "/preproc"
SRC_DIR = BASE_DIR + "/src"
FSL_DIR = "/well/woolrich/projects/software/fsl"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    setup_fsl(FSL_DIR)

    subjects = []
    for directory in sorted(
        glob(PREPROC_DIR + f"/sub*_task-{TASK}_proc-sss_meg")
    ):
        subject = Path(directory).name.split("_")[0]
        subjects.append(subject)

    # Find a good template subject to align other subjects to
    template = find_template_subject(
        SRC_DIR, subjects, n_embeddings=15, standardize=True
    )

    # Settings for batch processing
    config = f"""
        source_recon:
        - fix_sign_ambiguity:
            template: {template}
            n_embeddings: 15
            standardize: True
            n_init: 3
            n_iter: 3000
            max_flips: 20
    """

    # Setup parallel processing
    client = Client(n_workers=16, threads_per_worker=1)

    # Do the sign flipping
    run_src_batch(config, SRC_DIR, subjects, dask_client=True)
