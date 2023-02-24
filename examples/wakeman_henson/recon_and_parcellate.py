#!/usr/bin/env python

"""
Run group analysis on parcellated data on the Wakeman-Henson dataset.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>

import os
import os.path as op
from osl import source_recon
import numpy as np

subjects_dir = "/ohba/pi/mwoolrich/datasets/WakemanHenson/ds117"
out_dir = "./wakehen_glm"

subjects_dir = "/Users/woolrich/homedir/vols_data/WakeHen"
out_dir = op.join(subjects_dir, "wakehen_glm")

nsubjects = 19
nsessions = 6
subjects_to_do = np.arange(0, nsubjects)
sessions_to_do = np.arange(0, nsessions)
subj_sess_2exclude = np.zeros([nsubjects, nsessions]).astype(bool)

subj_sess_2exclude = np.ones(subj_sess_2exclude.shape).astype(bool)
subj_sess_2exclude[0:1,0:6]=False

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
            smri_file = op.join(subjects_dir, sub_name, "anatomy", "highres001.nii.gz")

            preproc_fif_file = op.join(
                subjects_dir, subject + "_meg", subject + "_meg_preproc_raw.fif"
            )

            # output files
            sflip_parc_file = op.join(recon_dir, subject, "sflip_parc.npy")

            if op.exists(preproc_fif_file) and op.exists(smri_file):
                subjects.append(subject)
                smri_files.append(smri_file)
                preproc_fif_files.append(preproc_fif_file)
                sflip_parc_files.append(sflip_parc_file)

# -------------------------------------------------------------
# %% Coreg and Source recon and Parcellate

config = """
    source_recon:
    - extract_fiducials_from_fif: {}
    - compute_surfaces_coregister_and_forward_model:
        include_nose: false
        use_nose: false
        use_headshape: false
        model: Single Layer
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: [mag, grad]
        rank: {meg: 58}
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
config = """
    source_recon:
    - fix_sign_ambiguity:
        template: {}
        n_embeddings: 15
        standardize: True
        n_init: 3
        n_iter: 2000
        max_flips: 20
""".format(template)

# Do the sign flipping
source_recon.run_src_batch(
    config,
    recon_dir,
    subjects,
)

# -------------------------------------------------------------
# %% Copy sf files to a single directory (makes it easier to copy minimal
# files to, e.g. BMRC, for downstream analysis)

os.makedirs(op.join(recon_dir, "sflip_data"), exist_ok=True)

for subject, sflip_parc_file in zip(subjects, sflip_parc_files):

    sflip_parc_file_to = op.join(
        recon_dir, "sflip_data", subject + "_sflip_parc.npy"
    )

    os.system("cp -f {} {}".format(sflip_parc_file, sflip_parc_file_to))

    sflip_parc_file_from = op.join(
        recon_dir, subject, "sflip_parc-raw.fif"
    )

    sflip_parc_file_to = op.join(
        recon_dir, "sflip_data", subject + "_sflip_parc-raw.fif"
    )

    os.system("cp -f {} {}".format(sflip_parc_file_from, sflip_parc_file_to))

####

if False:

    workshop_subjects_dir = '/Users/woolrich/CloudDocs/workshop/coreg_clean/data/wake_hen_group'
    workshop_recon_dir = op.join(workshop_subjects_dir, 'recon')

    os.system("mkdir {}".format(workshop_recon_dir))
    file_from = op.join(recon_dir, "report")
    file_to = op.join(workshop_recon_dir + '/')

    os.system("cp -fr {} {}".format(file_from, file_to))

    for subject in subjects:

        os.system("mkdir {}".format(op.join(workshop_recon_dir, subject)))

        file_from = op.join(recon_dir, subject, "sflip_parc-raw.fif")
        file_to = op.join(workshop_recon_dir, subject + '/')
        os.system("cp -f {} {}".format(file_from, file_to))

        subjects_dir, subject + "_meg", subject + "_meg_preproc_raw.fif"

        os.system("mkdir {}".format(op.join(workshop_subjects_dir, subject + "_meg")))

        file_from = op.join(subjects_dir, subject + "_meg/", "*.*")
        file_to = op.join(workshop_subjects_dir, subject + "_meg/")

        os.system("cp -f {} {}".format(file_from, file_to))
