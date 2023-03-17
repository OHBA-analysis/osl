"""Example script for sign flipping parcellated data from the LEMON dataset.

"""

# Authors : Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from glob import glob
from dask.distributed import Client

from osl import utils
from osl.source_recon import find_template_subject, run_src_batch, setup_fsl

# Directories
SRC_DIR = "/well/woolrich/projects/lemon/osl_example/src"
FSL_DIR = "/well/woolrich/projects/software/fsl"

# Subjects not to include (due to bad coregistration)
EXCLUDE = [
    "sub-010052", "sub-010061", "sub-010065", "sub-010070", "sub-010073",
    "sub-010074", "sub-010207", "sub-010287", "sub-010288",
]

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    setup_fsl(FSL_DIR)

    # Get subjects
    subjects = []
    for path in sorted(glob(SRC_DIR + "/*/rhino/parc-raw.fif")):
        subject = path.split("/")[-3]
        subjects.append(subject)

    # Remove subjects we don't want to sign flip
    for ex in EXCLUDE:
        if ex in subjects:
            subjects.remove(ex)

    # Find a good subject to align other subjects to
    template = find_template_subject(
        SRC_DIR, subjects, n_embeddings=15, standardize=True
    )

    # Settings
    config = f"""
        source_recon:
        - fix_sign_ambiguity:
            template: {template}
            n_embeddings: 15
            standardize: True
            n_init: 3
            n_iter: 5000
            max_flips: 7
    """

    # Setup parallel processing
    client = Client(n_workers=2, threads_per_worker=1)

    # Run sign flipping
    run_src_batch(config, SRC_DIR, subjects, dask_client=True)
