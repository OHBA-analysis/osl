"""Example script for extracting previously computed surfaces and coregistration.

Sometimes newer version of OSL aren't backwards compatible. This maybe an issue
if you want to use files created with an older version of OSL. This often happens
in the coregistration step, where you want to use previously computed surfaces
and coregistration.

In this script, we extract previously computed surfaces and the coregistation and
create a new directory which will work for the latest version of OSL.
"""

from osl.source_recon import setup_fsl
from osl.source_recon.rhino.utils import extract_rhino_files

setup_fsl("/opt/ohba/fsl/6.0.5")

old_dir = "path/to/old/src/dir"
new_dir = "src"

extract_rhino_files(old_dir, new_dir)
