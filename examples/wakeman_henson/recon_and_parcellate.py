#!/usr/bin/env python

"""
Run group analysis on parcellated data on the Wakeman-Henson dataset.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>

import os.path as op
from osl import source_recon
import numpy as np

subjects_dir = "/ohba/pi/mwoolrich/datasets/WakemanHenson/ds117"
out_dir = "./wakehen_glm"

subjects_dir = "/Users/woolrich/homedir/vols_data/WakeHen"
out_dir = op.join(subjects_dir, "wakehen_glm")

subjects_to_do = np.arange(0, 19)
sessions_to_do = np.arange(0, 6)
subj_sess_2exclude = np.zeros(subj_sess_2exclude.shape).astype(bool)

# -------------------------------------------------------------
# %% Setup file names

smri_files = []
preproc_fif_files = []
sflip_parc_files = []

subjects = []

recon_dir = op.join(subjects_dir, "recon")

for sub in subjects_to_do:
    for ses in sessions_to_do:
        if not subj_sess_2exclude[sub, ses]:

            sub_name = "sub" + ("{}".format(subjects_to_do[sub] + 1)).zfill(3)
            ses_name = "run_" + ("{}".format(sessions_to_do[ses] + 1)).zfill(2)
            subject = sub_name + "_" + ses_name

            # input files
            fif_in_file = op.join(subjects_dir, sub_name, "MEG", ses_name + "_sss.fif")
            smri_file = op.join(subjects_dir, sub_name, "anatomy", "highres001.nii.gz")

            preproc_fif_file = op.join(
                subjects_dir, subject + "_meg", subject + "_meg_preproc_raw.fif"
            )

            # output files
            sflip_parc_file = op.join(recon_dir, subject, "sflip_parc.npy")

            if op.exists(fif_in_file) and op.exists(smri_file):
                subjects.append(subject)
                smri_files.append(smri_file)
                preproc_fif_files.append(preproc_fif_file)
                sflip_parc_files.append(sflip_parc_file)

# -------------------------------------------------------------
# %% Coreg and Source recon and Parcellate

config = """
    source_recon:
    - extract_fiducials_from_fif: {}

    - coregister:
        include_nose: false
        use_nose: false
        use_headshape: true
        model: Single Layer
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: [mag, grad]
        rank: {meg: 60}
        parcellation_file: HarvOxf-sub-Schaefer100-combined-2mm_4d_ds8.nii.gz
        method: spatial_basis
        orthogonalisation: None
"""

# parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
# parcellation_file: Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_4d_ds8.nii.gz
# parcellation_file: HarvOxf-sub-Schaefer100-combined-2mm_4d_ds8.nii.gz

source_recon.run_src_batch(
    config,
    src_dir=recon_dir,
    subjects=subjects,
    preproc_files=preproc_fif_files,
    smri_files=smri_files,
)

# -------------------------------------------------------------
# %% Sign flip

# Find a good template subject to align other subjects to
template = source_recon.find_template_subject(
    recon_dir, subjects, n_embeddings=15, standardize=True
)

# Settings for batch processing
config = f"""
    source_recon:
    - fix_sign_ambiguity:
        template: {template}
        n_embeddings: 15
        standardize: True
        n_init: 3
        n_iter: 2500
        max_flips: 20
"""

# Do the sign flipping
source_recon.run_src_batch(
    config,
    recon_dir,
    subjects,
)

# -------------------------------------------------------------
# %% Copy sf files to a single directory (makes it easier to copy minimal
# files to, e.g. BMRC, for downstream analysis)

import os
os.makedirs(op.join(recon_dir, "sflip_data"), exist_ok=True)

for subject, sflip_parc_file in zip(subjects, sflip_parc_files):

    sflip_parc_file_to = op.join(
        recon_dir, "sflip_data", subject + "_sflip_parc.npy"
    )

    os.system("cp -f {} {}".format(sflip_parc_file, sflip_parc_file_to))