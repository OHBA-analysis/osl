"""Coregister with the electrodes as headshape points.

The montage's electrode positions rarely match the subject's head, so they
are rescaled (rescale_sensor_positions moves them in the preprocessed file)
before the coregistration. Adjust rescale until the electrodes sit on the
scalp in the coregistration plot of the report.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
coreg:
  - extract_surfaces: {include_nose: false}
  - rescale_sensor_positions: {rescale: [0.9, 1, 1]}
  - extract_fiducials_and_headshape: {include_eeg_as_headshape: true}
  - coregister: {use_nose: false, use_headshape: true}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=4)
