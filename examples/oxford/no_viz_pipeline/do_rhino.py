"""RHINO without any visualisations.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np

from osl.source_recon import setup_fsl, rhino

setup_fsl("/opt/ohba/fsl/6.0.5")  # this is where FSL is installed on hbaws

# Directories
preproc_dir = "data/preproc"
anat_dir = "data/smri"
coreg_dir = "data/coreg"

# Files ({subject} will be replaced by the name for the subject)
preproc_file = preproc_dir + "/{subject}_tsss_preproc_raw.fif"
smri_file = anat_dir + "/{subject}/anat/{subject}_T1w.nii"

# Subjects to coregister
subjects = ["sub-001", "sub-002"]

def fix_headshape_points(src_dir, subject, preproc_file, smri_file):
    filenames = rhino.get_coreg_filenames(src_dir, subject)

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
    np.savetxt(filenames["polhemus_headshape_file"], hs)


# Main RHINO code

for subject in subjects:

    # Input files
    preproc_file_ = preproc_file.format(subject=subject)
    smri_file_ = smri_file.format(subject=subject)

    # Extract fiducials and headshape points
    filenames = rhino.get_coreg_filenames(src_dir, subject)
    rhino.extract_polhemus_from_info(
        fif_file=preproc_file_,
        headshape_outfile=filenames["polhemus_headshape_file"],
        nasion_outfile=filenames["polhemus_nasion_file"],
        rpa_outfile=filenames["polhemus_rpa_file"],
        lpa_outfile=filenames["polhemus_lpa_file"],
    )

    # Clean up headshape points
    fix_headshape_points(src_dir, subject, preproc_file_, smri_file_)

    # Extract surfaces from the structural
    rhino.compute_surfaces(
        smri_file=smri_file_,
        subjects_dir=src_dir,
        subject=subject,
        include_nose=False,
    )

    # Coregistration
    rhino.coreg(
        fif_file=preproc_file_,
        subjects_dir=src_dir,
        subject=subject,
        use_headshape=True,
        use_nose=False,
    )

    # Forward model
    rhino.forward_model(
        subjects_dir=src_dir,
        subject=subject,
        model="Single Layer"
        gridstep=8,
    )
