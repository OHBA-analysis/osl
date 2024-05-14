"""Perform dipole sign flipping.

"""

import os
from glob import glob
from dask.distributed import Client

from osl import utils
from osl.source_recon import find_template_subject, run_src_batch

def run(cmd):
    print(cmd)
    os.system(cmd)

src_dir = "/well/woolrich/projects/mrc_meguk/all_sites/src"
sflip_dir = "/well/woolrich/projects/mrc_meguk/all_sites/sflip"

# Copy the parc-raw.fif files into one big directory
for path in sorted(glob(f"{src_dir}/*/*/parc/parc-raw.fif")):
    subject = path.split("/")[-3]
    run(f"mkdir -p {sflip_dir}/{subject}/parc")
    run(f"cp {path} {sflip_dir}/{subject}/parc")

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    subjects = []
    for path in sorted(glob(f"{sflip_dir}/*/parc/parc-raw.fif")):
        subject = path.split("/")[-3]
        subjects.append(subject)

    template = find_template_subject(
        sflip_dir, subjects, n_embeddings=15, standardize=True
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

    run_src_batch(
        config,
        src_dir=sflip_dir,
        subjects=subjects,
        dask_client=True,
    )
