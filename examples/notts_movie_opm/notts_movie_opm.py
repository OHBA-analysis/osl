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
from osl.source_recon import rhino
from osl.source_recon import parcellation
from osl.source_recon import beamforming
from osl.source_recon import batch

from osl import preprocessing

import yaml

from osl.utils import opm

subjects_to_do = np.arange(0, 10)
sessions_to_do = np.arange(0, 2)
subj_sess_2exclude = np.zeros([10, 2]).astype(bool)

#subj_sess_2exclude = np.ones([10, 2]).astype(bool)
#subj_sess_2exclude[0,:] = True

run_convert = False
run_preproc = False
run_compute_surfaces = False
run_coreg = True
run_forward_model = True
run_source_recon_parcellate = True
run_orth = True
run_extract_parcel_timeseries = True

# parcellation to use
parcellation_fname = op.join('/Users/woolrich/Dropbox/vols_scripts/hmm_misc_funcs/parcellations',
                             'fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz')

std_brain = '/Users/woolrich/homedir/vols_data/mne/self_paced_fingertap/subject1/rhino/surfaces/MNI152_T1_brain_2mm.nii.gz'

subjects_dir = '/Users/woolrich/homedir/vols_data/notts_movie_opm'

rank = {'mag': 100}
chantypes = ['mag']
freq_range = (1, 45)

# resolution of dipole grid for source recon
gridstep = 8  # mm

# -------------------------------------------------------------
# %% Setup file names

subjects = []
notts_opm_mat_files = []
smri_files = []
tsv_files = []

fif_files = []
preproc_fif_files = []
recon_dirs = []

for sub in subjects_to_do:
    for ses in sessions_to_do:
        if not subj_sess_2exclude[sub, ses]:
            sub_dir = 'sub-' + ('{}'.format(subjects_to_do[sub]+1)).zfill(3)
            ses_dir = 'ses-' + ('{}'.format(sessions_to_do[ses]+1)).zfill(3)
            subject = sub_dir + '_' + ses_dir

            # input files
            notts_opm_mat_file = op.join(subjects_dir, sub_dir, ses_dir, subject + '_meg.mat')
            smri_file = op.join(subjects_dir, sub_dir, 'mri', sub_dir + '.nii')
            tsv_file = op.join(subjects_dir, sub_dir, ses_dir, subject + '_channels.tsv')

            # output files
            fif_file = op.join(subjects_dir, subject, subject + '_meg.fif')
            recon_dir = op.join(subjects_dir, subject, 'meg')
            preproc_fif_file = op.join(recon_dir, subject + '_meg_preproc_raw.fif')

            # check opm file and structural file exists for this subject
            if op.exists(notts_opm_mat_file) and op.exists(smri_file):
                subjects.append(subject)
                notts_opm_mat_files.append(notts_opm_mat_file)
                smri_files.append(smri_file)
                tsv_files.append(tsv_file)

                fif_files.append(fif_file)
                preproc_fif_files.append(preproc_fif_file)
                recon_dirs.append(recon_dirs)

                # Make directories that will be needed
                if not os.path.isdir(op.join(subjects_dir, subject)):
                    os.mkdir(op.join(subjects_dir, subject))
                if not os.path.isdir(recon_dir):
                    os.mkdir(op.join(recon_dir))

# -------------------------------------------------------------
# %% Create fif files

if run_convert:
    for notts_opm_mat_file, tsv_file, fif_file in zip(notts_opm_mat_files, tsv_files, fif_files):
        opm.convert_notts(notts_opm_mat_file, tsv_file, fif_file)

# -------------------------------------------------------------
# %% Sort out structural

if False:
    smri_file_new = op.join(subjects_dir, sub_dir, 'mri', sub_dir +'_copy.nii.gz')

    # Copy smri_name to new file for modification
    copyfile(smri_file_in, smri_file_new)

    smri = nib.load(smri_file_new)
    sform = smri.header.get_sform()
    sform_std = np.copy(sform)
    sform_std[0, 0:4] = [1, 0, 0, -90]
    sform_std[1, 0:4] = [0, -1, 0, 126]
    sform_std[2, 0:4] = [0, 0, -1, 72]
    rhino.rhino_utils.system_call('fslorient -setsform {} {}'.format(' '.join(map(str, sform_std.flatten())), smri_file_new))

    smri_file = smri_file_new

# -------------------------------------------------------------
# %% Run preproc

if run_preproc:

    config_text = """
    meta:
      event_codes:

    preproc:
        - resample:     {sfreq: 150, n_jobs: 6}            
        - filter:       {l_freq: 1, h_freq: 45}
        - bad_segments: {segment_len: 800, picks: 'meg'}
        - bad_channels: {picks: 'meg'}        
    """

    # - bad_segments: {segment_len: 800, picks: 'meg'}
    # - bad_channels: {picks: 'meg'}
    config = yaml.load(config_text, Loader=yaml.FullLoader)

    # This outputs fif_file
    dataset = preprocessing.run_proc_batch(config, fif_files, outdir=subjects_dir, overwrite=True)

    for subject in subjects:
        os.system('mv {} {}'.format(
            op.join(subjects_dir, subject + '_meg_preproc_raw.*'),
            op.join(subjects_dir, subject, 'meg')
        ))


##########################

if run_compute_surfaces:

    rhino.compute_surfaces(smri_file,
                           subjects_dir, subject,
                           include_nose=False,
                           cleanup_files=False)

    rhino.surfaces_display(subjects_dir, subject)

##########################

if run_coreg:

    rhino.coreg(
        preproc_fif_file,
        subjects_dir,
        subject,
        already_coregistered=True)

    rhino.coreg_display(subjects_dir, subject,
                        plot_type='surf',
                        display_outskin=True,
                        display_outskin_with_nose=True,
                        display_fiducials=True,
                        display_sensors=True,
                        display_sensor_oris=True,
                        display_headshape_pnts=True)

###########################
#  Forward modelling

if run_forward_model:
    rhino.forward_model(subjects_dir, subject,
                        model='Single Layer',
                        gridstep=gridstep,
                        mindist=4.0)

    if False:
        rhino.bem_display(subjects_dir, subject,
                      plot_type='surf',
                      display_outskin_with_nose=False,
                      display_sensors=True,
                      display_sensor_oris=True)


    if False:
        # -------------------------------------------------------------
        # %% Take a look at leadfields

        # load forward solution
        fwd_fname = rhino.get_coreg_filenames(subjects_dir, subject)['forward_model_file']
        fwd = mne.read_forward_solution(fwd_fname)

        leadfield = fwd['sol']['data']
        print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

# -------------------------------------------------------------
# %% Source recon
