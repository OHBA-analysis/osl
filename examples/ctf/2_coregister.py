"""Coregistration and forward modelling.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
import pandas as pd

from osl import source_recon, utils

def save_polhemus_from_pos(src_dir, subject, preproc_file, smri_file, epoch_file):
    """Saves fiducials/headshape from a pos file."""

    # Get path to pos file
    pos_file = f"data/raw/Nottingham/{subject}/meg/{subject}_headshape.pos"
    utils.logger.log_or_print(f"Saving polhemus info from {pos_file}")

    # Get coreg filenames
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load in txt file, these values are in cm in polhemus space:
    num_headshape_pnts = int(pd.read_csv(pos_file, header=None).to_numpy()[0])
    data = pd.read_csv(pos_file, header=None, skiprows=[0], delim_whitespace=True)

    # RHINO is going to work with distances in mm
    # So convert to mm from cm, note that these are in polhemus space
    data.iloc[:, 1:4] = data.iloc[:, 1:4] * 10

    # Polhemus fiducial points in polhemus space
    polhemus_nasion = (
        data[data.iloc[:, 0].str.match("nasion")]
        .iloc[0, 1:4].to_numpy().astype("float64").T
    )
    polhemus_rpa = (
        data[data.iloc[:, 0].str.match("right")]
        .iloc[0, 1:4].to_numpy().astype("float64").T
    )
    polhemus_lpa = (
        data[data.iloc[:, 0].str.match("left")]
        .iloc[0, 1:4].to_numpy().astype("float64").T
    )

    # Polhemus headshape points in polhemus space in mm
    polhemus_headshape = (
        data[0:num_headshape_pnts]
        .iloc[:, 1:4].to_numpy().astype("float64").T
    )

    # Save
    np.savetxt(filenames["polhemus_nasion_file"], polhemus_nasion)
    np.savetxt(filenames["polhemus_rpa_file"], polhemus_rpa)
    np.savetxt(filenames["polhemus_lpa_file"], polhemus_lpa)
    np.savetxt(filenames["polhemus_headshape_file"], polhemus_headshape)

def fix_headshape_points(src_dir, subject, preproc_file, smri_file, epoch_file):
    """Remove headshape points on the nose and neck."""

    # Load saved headshape and nasion files
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Remove headshape points on the nose
    remove = np.logical_and(hs[1] > max(lpa[1], rpa[1]), hs[2] < nas[2])
    hs = hs[:, ~remove]

    # Remove headshape points on the neck
    remove = hs[2] < min(lpa[2], rpa[2]) - 4
    hs = hs[:, ~remove]

    # Remove headshape points far from the head in any direction
    remove = np.logical_or(
        hs[0] < lpa[0] - 5,
        np.logical_or(
            hs[0] > rpa[0] + 5,
            hs[1] > nas[1] + 5,
        ),
    )
    hs = hs[:, ~remove]

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)

# Settings
config = """
    source_recon:
    - save_polhemus_from_pos: {}
    - fix_headshape_points: {}
    - compute_surfaces:
        include_nose: False
    - coregister:
        use_nose: False
        use_headshape: True
    - forward_model:
        model: Single Layer
"""

# Subject IDs
subjects = ["sub-not001", "sub-not002"]

# Fif files containing the sensor-level preprocessed data for each subject
preproc_files = [
    "data/preproc/sub-not001_task-resteyesopen_meg/sub-not001_task-resteyesopen_meg_preproc_raw.fif",
    "data/preproc/sub-not002_task-resteyesopen_meg/sub-not002_task-resteyesopen_meg_preproc_raw.fif",
]

# The corresponding structurals for each subject
smri_files = [
    "data/smri/sub-not001_T1w.nii.gz",
    "data/smri/sub-not002_T1w.nii.gz",
]

# Directory to save output to
coreg_dir = "data/coreg"

# Source reconstruction
source_recon.run_src_batch(
    config,
    src_dir=coreg_dir,
    subjects=subjects,
    preproc_files=preproc_files,
    smri_files=smri_files,
    extra_funcs=[save_polhemus_from_pos, fix_headshape_points],
)
