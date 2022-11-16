"""Source reconstruction.

This include coregistration, beamforming, parcellation and orthogonalisation.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import source_recon


# Directories
RAW_DIR = "/ohba/pi/knobre/datasets/covid/rawbids"
PREPROC_DIR = "/ohba/pi/knobre/cgohil/covid/preproc"
SRC_DIR = "/ohba/pi/knobre/cgohil/covid/src"

# Files
SMRI_FILE = RAW_DIR + "/{0}/anat/{0}_T1w.nii"
PREPROC_FILE = PREPROC_DIR + "/{0}_task-restEO/{0}_task-restEO_preproc_raw.fif"

# Subjects to do
SUBJECTS = ["sub-004", "sub-005"]

# Settings
config = """
    source_recon:
    - extract_fiducials_from_fif: {}
    - coregister:
        include_nose: true
        use_nose: true
        use_headshape: true
        model: Single Layer
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
for subject in SUBJECTS:
    smri_files.append(SMRI_FILE.format(subject))
    preproc_files.append(PREPROC_FILE.format(subject))

# Source reconstruction
source_recon.run_src_batch(
    config,
    src_dir=SRC_DIR,
    subjects=SUBJECTS,
    preproc_files=preproc_files,
    smri_files=smri_files,
)
