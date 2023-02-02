"""Source reconstruction.

This include coregistration, beamforming, parcellation and orthogonalisation.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import source_recon


# Directories
raw_dir = "/ohba/pi/knobre/datasets/covid/rawbids"
preproc_dir = "/ohba/pi/knobre/cgohil/covid/preproc"
src_dir = "/ohba/pi/knobre/cgohil/covid/src"

# Files
smri_file = raw_dir + "/{0}/anat/{0}_T1w.nii"
preproc_file = preproc_dir + "/{0}_task-restEO/{0}_task-restEO_preproc_raw.fif"

# Subjects to do
subjects = ["sub-006", "sub-007"]

# Settings
config = """
    source_recon:
    - extract_fiducials_from_fif: {}
    - compute_surfaces_coregister_and_forward_model:
        include_nose: false
        use_nose: false
        use_headshape: true
        model: Single Layer
        recompute_surfaces: true
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: [mag, grad]
        rank: {meg: 60}
        parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

# Setup FSL (this is the directory on hbaws that contains FSL)
source_recon.setup_fsl("/opt/ohba/fsl/6.0.5")

# Get paths to files
smri_files = []
preproc_files = []
for subject in subjects:
    smri_files.append(smri_file.format(subject))
    preproc_files.append(preproc_file.format(subject))

# Source reconstruction
source_recon.run_src_batch(
    config,
    src_dir=src_dir,
    subjects=subjects,
    preproc_files=preproc_files,
    smri_files=smri_files,
)
