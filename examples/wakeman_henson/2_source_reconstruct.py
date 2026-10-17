"""Coregister, beamform and parcellate.

The headshape points on the nose are removed before the coregistration,
which does not use the nose. The parcel time courses are
data/<id>/<id>_parc-raw.fif.
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
  - beamform: {chantypes: [mag, grad], rank: {meg: 60}, frequency_range: [1, 45]}
parcellation:
  - parcellate:
      parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz
      orthogonalisation: symmetric
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=16)
