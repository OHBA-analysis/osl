"""Coregistration for source reconstruction.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import source_recon


RAW_DIR = "/ohba/pi/knobre/datasets/covid/rawbids"
PREPROC_DIR = "/ohba/pi/knobre/cgohil/covid/preproc"
SRC_DIR = "/ohba/pi/knobre/cgohil/covid/src"

SMRI_FILE = RAW_DIR + "/sub-{0}/anat/sub-{0}_T1w.nii"
PREPROC_FILE = PREPROC_DIR + "/sub-{0}_task-restEO_preproc_raw.fif"

SUBJECTS = ["004", "005"]

# Settings
config = """
    source_recon:
    - extract_fiducials_from_fif: {}
    - coregister:
        include_nose: true
        use_nose: true
        use_headshape: true
        model: Single Layer
"""

# Setup
source_recon.setup_fsl("/home/cgohil/local/fsl")

smri_files = []
preproc_files = []
for subject in SUBJECTS:
    smri_files.append(SMRI_FILE.format(subject))
    preproc_files.append(PREPROC_FILE.format(subject))

# Coregistration
source_recon.run_src_batch(
    config,
    src_dir=SRC_DIR,
    subjects=SUBJECTS,
    preproc_files=preproc_files,
    smri_files=smri_files,
)
