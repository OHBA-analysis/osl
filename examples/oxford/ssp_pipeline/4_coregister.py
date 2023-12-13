"""Coregisteration.

The scripts was first run for all subjects (with n_init=1). Then for subjects
whose coregistration looked a bit off we re-run this script just for that
particular subject with a higher n_init.

Note, these scripts do not include/use the nose in the coregistration.
If you want to use the nose you need to change the config to include the nose
and you may not want to call the fix_headshape_points function.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
from dask.distributed import Client

from osl import source_recon, utils

# Directories
preproc_dir = "output/preproc_ssp"
anat_dir = "path/to/smri/dir"
coreg_dir = "output/coreg"
fsl_dir = "/opt/ohba/fsl/6.0.5"  # this is where FSL is installed on hbaws

# Files ({subject} will be replaced by the name for the subject)
preproc_file = preproc_dir + "/{subject}_tsss_preproc_raw.fif"
smri_file = anat_dir + "/{subject}/anat/{subject}_T1w.nii"

# Subjects to coregister
subjects = ["sub-001", "sub-002"]

# Settings
config = """
    source_recon:
    - extract_fiducials_from_fif: {}
    - fix_headshape_points: {}
    - compute_surfaces:
        include_nose: False
    - coregister:
        use_nose: False
        use_headshape: True
        #n_init: 50
"""

def fix_headshape_points(src_dir, subject, preproc_file, smri_file, epoch_file):
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
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

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(fsl_dir)

    # Setup files
    preproc_files = []
    smri_files = []
    for subject in subjects:
        preproc_files.append(preproc_file.format(subject=subject))
        smri_files.append(smri_file.format(subject=subject))

    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=4, threads_per_worker=1)

    # Run coregistration
    source_recon.run_src_batch(
        config,
        src_dir=coreg_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_points],
        dask_client=True,
    )
