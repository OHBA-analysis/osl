"""Source reconstruction.

This includes beamforming, parcellation and orthogonalisation.

Note, before this script is run the /coreg directory created by 3_coregister.py
must be copied and renamed to /src.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from dask.distributed import Client

from osl import source_recon, utils

# Directories
preproc_dir = "output/preproc"
src_dir = "output/src"
fsl_dir = "/opt/ohba/fsl/6.0.5"  # this is where FSL is installed on hbaws

# Files
preproc_file = preproc_dir + "/{subject}_tsss_preproc_raw.fif"  # {subject} will be replaced by the subject name

# Subjects to do
subjects = ["sub-001", "sub-002"]

# Settings
config = """
    source_recon:
    - forward_model:
        model: Single Layer
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: [mag, grad]
        rank: {meg: 60}
        parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(fsl_dir)

    # Get paths to files
    preproc_files = []
    for subject in subjects:
        preproc_files.append(preproc_file.format(subject=subject))

    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=4, threads_per_worker=1)

    # Source reconstruction
    source_recon.run_src_batch(
        config,
        src_dir=src_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        dask_client=True,
    )
