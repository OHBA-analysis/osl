
"""Source reconstruction.

This includes: coregistration, beamforming and parcellation.
"""

import numpy as np
from dask.distributed import Client

from osl import source_recon, utils

source_recon.setup_fsl("/well/woolrich/projects/software/fsl")

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

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    config = """
        source_recon:
        - extract_fiducials_from_fif: {}
        - fix_headshape_points: {}
        - compute_surfaces_coregister_and_forward_model:
            include_nose: False
            use_nose: False
            use_headshape: True
            model: Single Layer
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: [mag, grad]
            rank: {meg: 60}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
    """

    raw_dir = f"/well/woolrich/projects/wakeman_henson/ds117"
    src_dir = "/well/woolrich/projects/wakeman_henson/spring23/src"

    subjects = []
    preproc_files = []
    smri_files = []
    for sub in range(1, 20):
        preproc_dir = f"/well/woolrich/projects/wakeman_henson/spring23/preproc/sub{sub:02d}"
        for run in range(1, 7):
            subjects.append(f"sub{sub:02d}_run{run:02d}")
            preproc_files.append(f"{preproc_dir}/run_{run:02d}_sss/run_{run:02d}_sss_preproc_raw.fif")
            smri_files.append(f"{raw_dir}/sub{sub:03d}/anatomy/highres001.nii.gz")

    source_recon.run_src_batch(
        config,
        src_dir=src_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_points],
        dask_client=True,
    )
