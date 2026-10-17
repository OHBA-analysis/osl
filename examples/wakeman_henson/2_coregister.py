"""Coregister the head to the MRI and compute the forward model.

The headshape points on the nose are removed before the coregistration,
which does not use the nose. Check the coregistration tab of
data/qc/report.html; rerun a session that looks off with a higher n_init.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
coreg:
  - extract_surfaces: {include_nose: false}
  - extract_fiducials_and_headshape: {}
  - remove_stray_headshape_points: {nose: true, neck: false, far: false}
  - coregister: {use_nose: false, use_headshape: true}
source_recon:
  - forward_model: {model: Single Layer}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=16)
