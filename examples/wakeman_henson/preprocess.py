#!/usr/bin/env python

"""
Run group analysis on parcellated data on the Wakeman-Henson dataset.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>

import os
import os.path as op
import numpy as np
import osl

subjects_dir = "/ohba/pi/mwoolrich/datasets/WakemanHenson/ds117"
out_dir = "./wakehen_glm"

subjects_dir = "/Users/woolrich/homedir/vols_data/WakeHen"
out_dir = op.join(subjects_dir, "wakehen_glm")

nsubjects = 19
nsessions = 6
subjects_to_do = np.arange(0, nsubjects)
sessions_to_do = np.arange(0, nsessions)
subj_sess_2exclude = np.zeros([nsubjects, nsessions]).astype(bool)

#subj_sess_2exclude = np.ones(subj_sess_2exclude.shape).astype(bool)
#subj_sess_2exclude[0:1,0:2]=False

# -------------------------------------------------------------
# %% Setup file names

smri_files = []
fif_files = []
preproc_fif_files = []
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

            # output files
            fif_file = op.join(
                subjects_dir, subject + "_meg.fif"
            )

            preproc_fif_file = op.join(
                subjects_dir, subject + "_meg", subject + "_meg_preproc_raw.fif"
            )

            if op.exists(fif_in_file) and op.exists(smri_file):
                subjects.append(subject)
                smri_files.append(smri_file)
                fif_files.append(fif_file)
                preproc_fif_files.append(preproc_fif_file)

                # Copy fif_in_file to subjects_dir
                os.system("cp -f {} {}".format(fif_in_file, fif_file))

# -------------
# Preprocessing

config = """
    meta:
      event_codes:
        famous/first: 5
        famous/immediate: 6
        famous/last: 7
        unfamiliar/first: 13
        unfamiliar/immediate: 14
        unfamiliar/last: 15
        scrambled/first: 17
        scrambled/immediate: 18
        scrambled/last: 19
    preproc:
      - find_events:       {min_duration: 0.005}
      - set_channel_types: {EEG062: eog, EEG062: eog, EEG063: ecg}
      - filter:            {l_freq: 1.1, h_freq: 100}
      - notch_filter:      {freqs: 50 100 150}
      - resample:          {sfreq: 150}
      - filter:            {l_freq: 1, h_freq: 30, method: iir, iir_params: {order: 5, btype: bandpass, ftype: butter}}
      - bad_channels: {picks: 'mag', significance_level: 0.1}
      - bad_channels: {picks: 'grad', significance_level: 0.1}
      - bad_segments: {segment_len: 200, picks: 'mag', significance_level: 0.1}
      - bad_segments: {segment_len: 200, picks: 'mag', significance_level: 0.1, mode: diff}    
      - bad_segments: {segment_len: 500, picks: 'mag', significance_level: 0.1}
      - bad_segments: {segment_len: 500, picks: 'mag', significance_level: 0.1, mode: diff}
      - bad_segments: {segment_len: 800, picks: 'mag', significance_level: 0.1}  
      - bad_segments: {segment_len: 800, picks: 'mag', significance_level: 0.1, mode: diff}  
      - bad_segments: {segment_len: 200, picks: 'grad', significance_level: 0.1}
      - bad_segments: {segment_len: 200, picks: 'grad', significance_level: 0.1, mode: diff}
      - bad_segments: {segment_len: 500, picks: 'grad', significance_level: 0.1}
      - bad_segments: {segment_len: 500, picks: 'grad', significance_level: 0.1, mode: diff}      
      - bad_segments: {segment_len: 800, picks: 'grad', significance_level: 0.1}  
      - bad_segments: {segment_len: 800, picks: 'grad', significance_level: 0.1, mode: diff}  
     
      
"""

# Run preprocessing
osl.preprocessing.run_proc_batch(
    config, fif_files, outdir=subjects_dir, overwrite=True
)

