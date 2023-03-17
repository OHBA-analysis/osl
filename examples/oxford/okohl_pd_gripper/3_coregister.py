"""Source reconstruct continuous sensor-level data.

Note, if you get the following error:

    Package 'open3d' is required for this function to run but cannot be imported.
    Please install it into your python environment to continue.

You need to install open3d:

    pip install open3d
"""

# Authors: Oliver Kohl <oliver.kohl@psych.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from osl import source_recon, utils

#%% Setup FSL

# This is the directory on hbaws that contains FSL
source_recon.setup_fsl("/opt/ohba/fsl/6.0.5")

#%% Custom function

def fix_headshape_points(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    epoch_file,
    save_hs=False,
    make_plot=False,
):
    """Fix headshape points."""

    # Get coreg filenames
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Remove Headshape points in neck area
    if subject in [
        "HC01", "HC02", "HC05", "HC06", "HC08", "HC09", "HC12", "HC13",
        "HC14", "HC15", "HC16", "HC18", "HC19", "HC20", "HC22", "HC24",
        "PD01", "PD03", "PD04", "PD05", "PD06", "PD08", "PD09", "PD10",
        "PD13", "PD14", "PD16", "PD17", "PD18", "PD20", "PD22", "PD23",
        "PD28", "PD31",
    ]:
        higher_pa = np.max([lpa[2], rpa[2]])
        remove = np.logical_and(hs[1] < rpa[1], hs[2] < higher_pa) # , hs_cl[1] < lpa[1],hs_cl[2] < lpa[2])
        hs_removed = hs[:, remove]
        hs_cl = hs[:, ~remove]

    if subject in ["PD05"]:
        hs_tmp = hs_cl

        # Remove anything outside of rpa
        keep_r = hs_cl[0] < rpa[0]
        rm_r = ~(hs_tmp[0] < rpa[0])
        hs_cl = hs_cl[:, keep_r]

        # Remove anything outside of lpa
        keep_l = hs_cl[0] > lpa[0]
        rm_l = ~(hs_tmp[0] > lpa[0])
        hs_cl = hs_cl[:, keep_l]

        # Remove Headshape points that are too high
        keep_h = hs_cl[2] < 135
        rm_h = ~(hs_tmp[2] < 135)
        hs_cl = hs_cl[:, keep_h]

        remove = np.sum([rm_r, rm_l, rm_h], axis=0) > 0
        hs_removed = np.hstack([hs_removed, hs_tmp[:, remove]])

    if subject in ["HC24"]:
        # Remove anything outside of rpa
        keep_r = hs_cl[1] > -100
        remove = ~(hs_cl[1] > -100)
        hs_removed = np.hstack([hs_removed, hs_cl[:, remove]])
        hs_cl = hs_cl[:, keep_r]

    if make_plot:
        # Make Plot cleaned
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(projection="3d")
        ax.scatter3D(hs_cl[0], hs_cl[1], hs_cl[2], color="blue")
        ax.scatter3D(nas[0], nas[1], nas[2], color="black", s=18)
        ax.scatter3D(lpa[0], lpa[1], lpa[2], color="black", s=18)
        ax.scatter3D(rpa[0], rpa[1], rpa[2], color="black", s=18)
        ax.scatter3D(hs_removed[0], hs_removed[1], hs_removed[2], color="red", s=18)
        ax.set_title(subject, pad=25, size=15)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs_cl)

#%% Specify subjects and setup file paths

# Subjects to coregister
subjects = ["HC01", "HC02"]

# Directories
preproc_dir = "/ohba/pi/knobre/cgohil/pd_gripper/preproc"
smri_dir = "/ohba/pi/knobre/PD_data/MRI/NZ_data"
src_dir = "/ohba/pi/knobre/cgohil/pd_gripper/src"

# Setup paths to preprocessed data and structurals
preproc_files = []
smri_files = []
for subject in subjects:
    preproc_files.append(f"{preproc_dir}/{subject}_gripper_trans/{subject}_gripper_trans_preproc_raw.fif")
    smri_files.append(f"{smri_dir}/{subject}_Structural.nii")

#%% Run coregistration

# Settings
config = """
    source_recon:
    - extract_fiducials_from_fif: {}
    - fix_headshape_points:  # will overwrite headshape point from extract_fiducials_from_fif
        save_hs: false
        make_plot: false
    - compute_surfaces:
        include_nose: true
    - coregister:
        use_nose: true
        use_headshape: true
"""

# Run source reconstruction
source_recon.run_src_batch(
    config,
    src_dir=src_dir,
    subjects=subjects,
    preproc_files=preproc_files,
    smri_files=smri_files,
    extra_funcs=[fix_headshape_points],
)
