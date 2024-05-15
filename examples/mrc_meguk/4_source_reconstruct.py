"""Source reconstruction.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

# Elekta
do_oxford = False
do_cambridge = False

# CTF
do_nottingham = True
do_cardiff = True

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    if do_oxford:
        config = """
            source_recon:
            - forward_model:
                model: Single Layer
            - beamform_and_parcellate:
                freq_range: [1, 80]
                chantypes: [mag, grad]
                rank: {meg: 60}
                parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
                method: spatial_basis
                orthogonalisation: symmetric
        """

        preproc_dir = "/well/woolrich/projects/mrc_meguk/all_sites/preproc/Oxford"
        src_dir = "/well/woolrich/projects/mrc_meguk/all_sites/src/Oxford"

        subjects = []
        preproc_files = []
        for directory in sorted(glob(f"{src_dir}/*")):
            subject = directory.split("/")[-1]
            if "sub" not in subject:
                continue
            preproc_file = (
                f"{preproc_dir}/{subject}_proc-sss_meg"
                f"/{subject}_proc-sss_meg_preproc_raw.fif"
            )
            subjects.append(subject)
            preproc_files.append(preproc_file)

        source_recon.run_src_batch(
            config,
            src_dir=src_dir,
            subjects=subjects,
            preproc_files=preproc_files,
            dask_client=True,
        )

    if do_cambridge:
        config = """
            source_recon:
            - forward_model:
                model: Single Layer
            - beamform_and_parcellate:
                freq_range: [1, 80]
                chantypes: [mag, grad]
                rank: {meg: 60}
                parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
                method: spatial_basis
                orthogonalisation: symmetric
        """

        preproc_dir = "/well/woolrich/projects/mrc_meguk/all_sites/preproc/Cambridge"
        src_dir = "/well/woolrich/projects/mrc_meguk/all_sites/src/Cambridge"

        subjects = []
        preproc_files = []
        for directory in sorted(glob(f"{src_dir}/*")):
            subject = directory.split("/")[-1]
            if "sub" not in subject:
                continue
            preproc_file = (
                f"{preproc_dir}/{subject}_proc-sss_meg"
                f"/{subject}_proc-sss_meg_preproc_raw.fif"
            )
            subjects.append(subject)
            preproc_files.append(preproc_file)

        source_recon.run_src_batch(
            config,
            src_dir=src_dir,
            subjects=subjects,
            preproc_files=preproc_files,
            dask_client=True,
        )

    if do_nottingham:
        config = """
            source_recon:
            - forward_model:
                model: Single Layer
            - beamform_and_parcellate:
                freq_range: [1, 80]
                chantypes: mag
                rank: {mag: 120}
                parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
                method: spatial_basis
                orthogonalisation: symmetric
        """

        preproc_dir = "/well/woolrich/projects/mrc_meguk/all_sites/preproc/Nottingham"
        src_dir = "/well/woolrich/projects/mrc_meguk/all_sites/src/Nottingham"

        subjects = []
        preproc_files = []
        for directory in sorted(glob(f"{src_dir}/*")):
            subject = directory.split("/")[-1]
            if "sub" not in subject:
                continue
            preproc_file = f"{preproc_dir}/{subject}_meg/{subject}_meg_preproc_raw.fif"
            subjects.append(subject)
            preproc_files.append(preproc_file)

        source_recon.run_src_batch(
            config,
            src_dir=src_dir,
            subjects=subjects,
            preproc_files=preproc_files,
            dask_client=True,
        )

    if do_cardiff:
        config = """
            source_recon:
            - forward_model:
                model: Single Layer
            - beamform_and_parcellate:
                freq_range: [1, 80]
                chantypes: mag
                rank: {mag: 120}
                parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
                method: spatial_basis
                orthogonalisation: symmetric
        """

        preproc_dir = "/well/woolrich/projects/mrc_meguk/all_sites/preproc/Cardiff"
        src_dir = "/well/woolrich/projects/mrc_meguk/all_sites/src/Cardiff"

        subjects = []
        preproc_files = []
        for directory in sorted(glob(f"{src_dir}/*")):
            subject = directory.split("/")[-1]
            if "sub" not in subject:
                continue
            preproc_file = f"{preproc_dir}/{subject}_meg/{subject}_meg_preproc_raw.fif"
            subjects.append(subject)
            preproc_files.append(preproc_file)

        source_recon.run_src_batch(
            config,
            src_dir=src_dir,
            subjects=subjects,
            preproc_files=preproc_files,
            dask_client=True,
        )
