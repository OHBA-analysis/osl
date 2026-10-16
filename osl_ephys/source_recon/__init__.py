"""Source reconstruction: coregistration, forward modelling, LCMV
beamforming and minimum norm, parcellation and sign flipping.

Each stage reads and writes the files held by a
:class:`~osl_ephys.session.Session`; the config steps are in
:mod:`~osl_ephys.source_recon.steps`.
"""

from .batch import *

from .freesurfer import (
    setup_freesurfer,
    check_freesurfer,
    recon_all,
    make_watershed_bem,
    fsaverage_source_space,
)

# The config steps register themselves on import
from osl_ephys.pipeline import registry as _registry
from osl_ephys.source_recon import steps
