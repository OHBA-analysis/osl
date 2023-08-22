"""Example script for extracting previously computed surfaces and coregistration.

Sometimes newer version of OSL aren't backwards compatible. This maybe an issue
if you want to use files created with an older version of OSL. This often happens
in the coregistration step, where you want to use previously computed surfaces
and coregistration.

In this script, we extract previously computed surfaces and the coregistation and
create a new directory which will work for the latest version of OSL.

We give example code for doing this in serial and in parallel using the batch
processing - you don't need to do both.
"""

# ---------
# In serial

from osl.source_recon import setup_fsl
from osl.source_recon.rhino.utils import extract_rhino_files

setup_fsl("/opt/ohba/fsl/6.0.5")

old_dir = "path/to/old/src/dir"
new_dir = "src"

extract_rhino_files(old_dir, new_dir)

# -----------
# In parallel

from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    source_recon.setup_fsl("/opt/ohba/fsl/6.0.5")

    old_dir = "path/to/old/src/dir"
    new_dir = "src"

    subjects = [path.split("/")[-1] for path in sorted(glob(f"{old_dir}/sub-*"))]

    config = f"""
        source_recon:
        - extract_rhino_files: {{old_src_dir: {old_dir}}}
    """

    client = Client(n_workers=16, threads_per_worker=1)

    source_recon.run_src_batch(
        config,
        src_dir=new_dir,
        subjects=subjects,
        dask_client=True,
    )
