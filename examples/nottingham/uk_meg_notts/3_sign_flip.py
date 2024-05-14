"""Sign flip parcellated uk_meg_notts data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from glob import glob
from dask.distributed import Client

from osl import source_recon, utils
from osl.source_recon import find_template_subject, run_src_batch

SRC_DIR = "/well/woolrich/projects/uk_meg_notts/eo/oslpy22/src"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    subjects = []
    for path in sorted(glob(SRC_DIR + "/*/rhino/parc-raw.fif")):
        subject = path.split("/")[-3]
        subjects.append(subject)

    template = find_template_subject(
        SRC_DIR, subjects, n_embeddings=15, standardize=True
    )

    config = f"""
        source_recon:
        - fix_sign_ambiguity:
            template: {template}
            n_embeddings: 15
            standardize: True
            n_init: 3
            n_iter: 2500
            max_flips: 20
    """

    client = Client(n_workers=16, threads_per_worker=1)
    run_src_batch(config, SRC_DIR, subjects, dask_client=True)
