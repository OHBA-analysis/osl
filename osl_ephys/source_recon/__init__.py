from .batch import *  # noqa: F401, F403
from .rhino.fsl_utils import setup_fsl, check_fsl  # noqa: F401, F403
from .freesurfer_utils import (
    setup_freesurfer,
    check_freesurfer,
    recon_all,
    make_watershed_bem,
    make_fsaverage_src,
)  # noqa: F401, F403
from .wrappers import find_template_subject  # noqa: F401, F403

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    __doc__ = f.read()

# The v2 wrappers, registered in the sections they belong to with the v2
# calling convention, until phase 4 replaces them. find_template_subject is
# called from Python before a sign-flipping run, not as a step.
from osl_ephys.pipeline import registry as _registry

_SECTIONS = {
    "coreg": [
        "extract_polhemus_from_info",
        "extract_fiducials_from_fif",
        "extract_polhemus_from_pos",
        "extract_polhemus_from_elc",
        "remove_stray_headshape_points",
        "save_mni_fiducials",
        "rescale_sensor_positions",
        "compute_surfaces",
        "make_watershed_bem",
        "coregister",
        "extract_rhino_files",
    ],
    "source_recon": ["forward_model", "beamform", "minimum_norm"],
    "parcellation": [
        "parcellate",
        "beamform_and_parcellate",
        "minimum_norm_and_parcellate",
        "fix_sign_ambiguity",
    ],
}
for _section, _names in _SECTIONS.items():
    for _name in _names:
        _registry.register(getattr(wrappers, _name), _section, kind="legacy_src")
