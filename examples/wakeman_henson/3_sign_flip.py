"""Fix the dipole sign ambiguity.

"""

from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

# Directory containing source reconstructed data
src_dir = "/well/woolrich/projects/wakeman_henson/spring23/src"
src_files = sorted(glob(src_dir + "/*/parc/parc-raw.fif"))

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Get subjects
    subjects = []
    for path in src_files:
        subject = path.split("/")[-3]
        subjects.append(subject)

    # Find a good template subject to match others to
    template = source_recon.find_template_subject(
        src_dir, subjects, n_embeddings=15, standardize=True
    )

    # Settings
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

    # Run sign flipping
    source_recon.run_src_batch(config, src_dir, subjects, dask_client=True)
