#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:39:24 2021

@author: woolrich
"""

import os
import os.path as op

import numpy as np
import matplotlib.pyplot as plt

from osl import preprocessing
from osl import source_recon

import yaml

from osl.utils import opm

subjects_to_do = np.arange(0, 10)
sessions_to_do = np.arange(0, 2)
subj_sess_2exclude = np.zeros([10, 2]).astype(bool)

#subj_sess_2exclude = np.ones([10, 2]).astype(bool)
#subj_sess_2exclude[1:3, 0] = False

run_convert = True
run_preproc = True
run_beamform_and_parcellate = True
run_fix_sign_ambiguity = True

# parcellation to use
parcellation_fname = op.join('/Users/woolrich/Dropbox/vols_scripts/hmm_misc_funcs/parcellations',
                             'fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz')

std_brain = '/Users/woolrich/homedir/vols_data/self_paced_fingertap/subject1/rhino/surfaces/MNI152_T1_brain_2mm.nii.gz'

subjects_dir = '/Users/woolrich/homedir/vols_data/notts_movie_opm'

rank = {'mag': 100}
chantypes = ['mag']
freq_range = (1, 45)

# resolution of dipole grid for source recon
gridstep = 8  # mm

# -------------------------------------------------------------
# %% Setup file names

subjects = []
sub_dirs = []
notts_opm_mat_files = []
smri_files = []
smri_fixed_files = []
tsv_files = []

fif_files = []
preproc_fif_files = []

recon_dir = op.join(subjects_dir, 'recon')

for sub in subjects_to_do:
    for ses in sessions_to_do:
        if not subj_sess_2exclude[sub, ses]:

            sub_dir = 'sub-' + ('{}'.format(subjects_to_do[sub] + 1)).zfill(3)
            ses_dir = 'ses-' + ('{}'.format(sessions_to_do[ses] + 1)).zfill(3)
            subject = sub_dir + '_' + ses_dir

            # input files
            notts_opm_mat_file = op.join(subjects_dir, sub_dir, ses_dir, subject + '_meg.mat')
            smri_file = op.join(subjects_dir, sub_dir, 'mri', sub_dir + '.nii')
            smri_fixed_file = op.join(subjects_dir, sub_dir, 'mri', sub_dir + '_fixed.nii')
            tsv_file = op.join(subjects_dir, sub_dir, ses_dir, subject + '_channels.tsv')

            # output files
            fif_file = op.join(subjects_dir, subject, subject + '_meg.fif')
            preproc_fif_file = op.join(subjects_dir, subject, subject + '_meg_preproc_raw.fif')

            # check opm file and structural file exists for this subject
            if op.exists(notts_opm_mat_file) and op.exists(smri_file):
                subjects.append(subject)
                sub_dirs.append(sub_dir)
                notts_opm_mat_files.append(notts_opm_mat_file)
                smri_files.append(smri_file)

                smri_fixed_files.append(smri_fixed_file)
                tsv_files.append(tsv_file)
                fif_files.append(fif_file)
                preproc_fif_files.append(preproc_fif_file)

                # Make directories that will be needed
                if not os.path.isdir(op.join(subjects_dir, subject)):
                    os.mkdir(op.join(subjects_dir, subject))

# -------------------------------------------------------------
# %% Create fif files

if run_convert:
    for notts_opm_mat_file, tsv_file, fif_file, smri_file, smri_fixed_file in zip(notts_opm_mat_files, tsv_files,
                                                                                  fif_files, smri_files,
                                                                                  smri_fixed_files):
        opm.convert_notts(notts_opm_mat_file, smri_file, tsv_file, fif_file, smri_fixed_file)

smri_files = smri_fixed_files

# -------------------------------------------------------------
# %% Run preproc

if run_preproc:

    config_text = """
    meta:
      event_codes:

    preproc:
        - crop:         {tmin: 5}
        - resample:     {sfreq: 150, n_jobs: 6}            
        - filter:       {l_freq: 4, h_freq: 45}
        - bad_channels: {picks: 'meg'}        
    """

    config = yaml.load(config_text, Loader=yaml.FullLoader)

    dataset = preprocessing.run_proc_batch(config, fif_files, outdir=subjects_dir, overwrite=True)

    # preprocessing.run_proc_batch will output preproc fif_files in subjects_dir
    # we will now move them into the subjects_dir/subject dirs
    for subject, preproc_fif_file in zip(subjects, preproc_fif_files):
        os.system('mv {} {}'.format(
            op.join(subjects_dir, subject + '_meg_preproc_raw.fif'),
            preproc_fif_file
        ))


if False:
    # to just run coreg for subject 0:
    source_recon.rhino.coreg(
        preproc_fif_files[0],
        recon_dir,
        subjects[0],
        already_coregistered=True
    )

    # to view coreg result for subject 0:
    source_recon.rhino.coreg_display(recon_dir, subjects[2],
                                     plot_type='surf')

    # to just run surfaces for subject 0:
    source_recon.rhino.compute_surfaces(
        smri_files[0],
        recon_dir,
        subjects[0],
        overwrite=True
    )

    # to view surfaces for subject 0:
    source_recon.rhino.surfaces_display(recon_dir, subjects[3])

# -------------------------------------------------------------
# %% Coreg and Source recon and Parcellate

if run_beamform_and_parcellate:
    # Settings
    config = """
        source_recon:
        - coregister:
            include_nose: false
            use_nose: false
            use_headshape: true
            model: Single Layer
            already_coregistered: true
            overwrite: false
        - beamform_and_parcellate:
            freq_range: [5, 40]
            chantypes: mag
            rank: {mag: 100}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
    """

    source_recon.run_src_batch(
        config,
        src_dir=recon_dir,
        subjects=subjects,
        preproc_files=preproc_fif_files,
        smri_files=smri_files,
    )

if False:
    # -------------------------------------------------------------
    # %% Take a look at leadfields

    # load forward solution
    fwd_fname = rhino.get_coreg_filenames(subjects_dir, subject)['forward_model_file']
    fwd = mne.read_forward_solution(fwd_fname)

    leadfield = fwd['sol']['data']
    print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

# -------------------------------------------------------------
# %% Sign flip

if run_fix_sign_ambiguity:
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
    source_recon.run_src_batch(config, recon_dir, subjects)

