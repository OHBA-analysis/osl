import os

from .batch import *
from ._v2_rhino.fsl_utils import setup_fsl, check_fsl
from .freesurfer_utils import (
    setup_freesurfer,
    check_freesurfer,
    recon_all,
    make_watershed_bem,
    make_fsaverage_src,
)
from .wrappers import find_template_subject

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    __doc__ = f.read()

# The config steps register themselves on import; the v2 wrappers that have
# no replacement yet keep the v2 calling convention
from osl_ephys.pipeline import registry as _registry
from osl_ephys.source_recon import steps

for _section, _name in (
    ("coreg", "make_watershed_bem"),
    ("coreg", "extract_rhino_files"),
    ("source_recon", "minimum_norm"),
    ("parcellation", "minimum_norm_and_parcellate"),
):
    _registry.register(getattr(wrappers, _name), _section, kind="legacy_src")
