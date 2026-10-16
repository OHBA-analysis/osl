import os

from .batch import *
from .rhino import setup_fsl, check_fsl, fsleyes, fsleyes_overlay
from .freesurfer import (
    setup_freesurfer,
    check_freesurfer,
    recon_all,
    make_watershed_bem,
    fsaverage_source_space,
)

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    __doc__ = f.read()

# The config steps register themselves on import
from osl_ephys.pipeline import registry as _registry
from osl_ephys.source_recon import steps
