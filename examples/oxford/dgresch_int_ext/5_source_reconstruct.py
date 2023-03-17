"""Example script for source reconstructing epoched data.

Note, the parcellated data is saved as an MNE Epochs object: {src_dir}/*/{event_type}/rhino/parc-epo.fif
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import source_recon

# Setup paths to epoched data files
event_type = "internal_disp"
epoch_dir = f"/ohba/pi/knobre/cgohil/dg_int_ext/epoch/{event_type}"
subjects = ["s01_block_01", "s01_block_02"]

run_ids = []
preproc_files = []
epoch_files = []
for subject in subjects:
    run_id = f"InEx_{subject}_tsss"
    run_ids.append(run_id)
    preproc_files.append(f"{epoch_dir}/{run_id}_preproc_raw/{run_id}_preproc_raw.fif")
    epoch_files.append(f"{epoch_dir}/{run_id}_preproc_raw/{run_id}_preproc_raw_epo.fif")

# Setup paths to structural MRI files
smri_dir = "/ohba/pi/knobre/dgresch/Internal_External/data/mri/02_onlyT1"

smri_files = []
for subject in subjects:
    s = subject.split("_")[0]
    smri_files.append(f"{smri_dir}/{s}.nii")

# Settings
config = """
    source_recon:
    - extract_fiducials_from_fif: {}
    - compute_surfaces_coregister_and_forward_model:
        include_nose: false
        use_nose: false
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

# Output directory
src_dir = f"/ohba/pi/knobre/cgohil/dg_int_ext/src/{event_type}"

# Run source reconstruction
source_recon.run_src_batch(
    config,
    src_dir=src_dir,
    subjects=subjects,
    preproc_files=preproc_files,
    smri_files=smri_files,
    epoch_files=epoch_files,
)
