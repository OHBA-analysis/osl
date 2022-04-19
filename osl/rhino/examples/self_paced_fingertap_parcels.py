# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:39:24 2021

Runs parcellation based GLM analysis on Notts self-paced finger tap data

@author: woolrich
"""

import os
import os.path as op

import numpy as np
import matplotlib.pyplot as plt
from osl import rhino
from osl import parcellation

import glmtools

import mne

subjects_dir = '/Users/woolrich/homedir/vols_data/mne/self_paced_fingertap'
subject = 'subject1'

# load precomputed forward solution
fwd_fname = rhino.get_coreg_filenames(subjects_dir, subject)['forward_model_file']
fwd = mne.read_forward_solution(fwd_fname)

# preprocessed fif file
fif_file = op.join(subjects_dir, subject, 'JRH_MotorCon_20100429_01_FORMARK_raw.fif')

# parcellation to use
parcellation_fname = op.join('/Users/woolrich/Dropbox/vols_scripts/hmm_misc_funcs/parcellations',
                             'fmri_d100_parcellation_with_PCC_reduced_2mm.nii.gz')

# nii volume to overlay parcellation results on
parcellation_background_fname = op.join(os.environ['FSLDIR'], 'data/standard/MNI152_T1_2mm_brain.nii.gz')

run_recon = True
orthogonalise_parcel_timeseries = False

rank = {'mag': 125}
chantypes = ['mag']

# -------------------------------------------------------------
# Do source recon

recon_dir = op.join(subjects_dir, subject, 'rhino', 'recon')
if not os.path.isdir(recon_dir):
    os.mkdir(recon_dir)

####
# Get and setup the data

raw = mne.io.read_raw_fif(fif_file)

# Use MEG sensors
raw.pick(chantypes)

raw.load_data()

# focus on beta band
raw.filter(l_freq=13, h_freq=30, method='iir', iir_params={'order': 5, 'btype': 'bandpass', 'ftype': 'butter'})

# Use time window that excludes the initial rest period
time_from = 300  # secs
time_to = 1439.9933  # secs
raw.crop(time_from, time_to).load_data()

# Do hilbert transform
original_raw = raw.copy()
raw.apply_hilbert()

if run_recon:
    # make LCMV filter
    filters = rhino.make_lcmv(subjects_dir, subject,
                                original_raw,
                                chantypes,
                                reg=0,
                                pick_ori='max-power-pre-weight-norm',
                                weight_norm='nai',
                                rank=rank,
                                reduce_rank=True,
                                verbose=True)


    # stc is source space time series (in head/polhemus space)
    stc = mne.beamformer.apply_lcmv_raw(raw, filters, max_ori_out='signed')

    # hilbert transform gave us complex data, we want the amplitude:
    data = np.abs(stc.data)

    # Convert from head/polhemus space to standard brain grid in MNI space
    recon_timeseries_mni, reference_brain_fname, recon_coords_mni, _ = rhino.transform_recon_timeseries(
        subjects_dir, subject,
        recon_timeseries=data,
        reference_brain='mni')

    np.save(op.join(recon_dir, 'recon_timeseries_mni.npy'), recon_timeseries_mni)
    np.save(op.join(recon_dir, 'recon_coords_mni.npy'), recon_coords_mni)

recon_timeseries_mni = np.load(op.join(recon_dir, 'recon_timeseries_mni.npy'))
recon_coords_mni = np.load(op.join(recon_dir, 'recon_coords_mni.npy'))

# plot centre of mass for each parcel
p = parcellation.Parcellation(parcellation_fname)
p.plot()

# view parcellation in fsleyes
# note that it is a 4D niftii file, where the 4th dimension is over parcels

rhino.fsleyes_overlay(parcellation_background_fname, p.file)

# Apply parcellation to voxelwise data (voxels x tpts) contained in recon_timeseries_mni
# Resulting parcel_timeseries will be (parcels x tpts) in MNI space
parcel_timeseries = p.parcellate(recon_timeseries_mni, recon_coords_mni, method='spatial_basis')

# -------------------------------------------------------------
# %% Orthogonalise

if orthogonalise_parcel_timeseries:
    # this is nsources x ntpts x ntrials
    timeseries = parcel_timeseries['data']
    ortho_timeseries = parcellation.symmetric_orthogonalise(timeseries, True)

    # plot first bit of example parcel time course
    parcel_index = 20
    plt.figure()
    plt.plot(raw.times[:500], parcel_timeseries['data'][parcel_index, :500])
    plt.plot(raw.times[:500], ortho_timeseries[parcel_index, :500])
    plt.xlabel('time (s)')
    plt.legend(('Before orth', 'After orth'))

    # plot between parcel correlations before and after orthog
    plt.figure()
    fig, axs = plt.subplots(1, 2, sharey='row')
    axs[0].imshow(np.corrcoef(np.reshape(parcel_timeseries['data'],(39, -1))))
    axs[1].imshow(np.corrcoef(np.reshape(ortho_timeseries,(39, -1))))
    axs[0].title.set_text('Corrs before orthogonalisation')
    axs[1].title.set_text('Corrs after orthogonalisation')
    axs[0].set_xlabel('Parcel')
    axs[0].set_ylabel('Parcel')
    axs[1].set_xlabel('Parcel')

    # so that orth time series get used in rest of the script:
    parcel_timeseries['data'] = ortho_timeseries

# -------------------------------------------------------------
# %% Compute the power for each parcel and view as a niftii in fsleyes

parcel_power = np.mean(parcel_timeseries['data'], axis=1)/np.std(parcel_timeseries['data'], axis=1)
rhino.fsleyes_overlay(parcellation_background_fname, p.nii(parcel_power, method='assignments'))

# -------------------------------------------------------------
# %% Fit GLM to parcel time courses

# Establish design matrix
#
# As this experiment consists of a sequence of blocks of sustained
# motor tasks, instead of epoching and doing a trial-wise GLM, we are going
# to do a time-wise GLM (like you would in traditional task fMRI analysis).
#
# We need to generate the regressors to go in our design matrix for the GLM.
# The experiment is made up of sequences of finger tapping blocks.
# There were four types of blocks (conditions):
# 1. Left hand tap
# 2. Right hand tap
# 3. Rest
# 4. Both hands tap
# Plus:
# 5. A period of rest at the start
#
# The order of these blocks is specified below by block_order and their length
# in seconds is block_length.

# get time indices that correspond to the time window that was source
# reconstructed
ntotal_tpts = mne.io.read_raw_fif(fif_file).n_times

tres = 1 / raw.info['sfreq']
timepnts = np.arange(0, tres * ntotal_tpts, step=tres)
time_inds = np.where((timepnts >= time_from) & (timepnts <= (time_to + tres)))[0]
times = timepnts[time_inds]

block_length = tres * int(30 / tres)  # secs
block_order = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                        4, 3, 2, 1, 2, 3, 1, 4, 3, 4,
                        1, 3, 2, 1, 4, 4, 2, 1, 3, 3,
                        4, 1, 4, 3, 1, 2, 1, 2, 3, 4,
                        3, 4, 1, 2, 3, 4, 1, 2])

# Create the design matrix:
design_matrix = np.zeros([ntotal_tpts, 5])

tim = 0
for tt in range(len(block_order)):
    design_matrix[tim:int(tim + block_length / tres), block_order[tt] - 1] = 1
    tim += int(block_length / tres)

tim_crop = timepnts[time_inds]
design_matrix_crop = design_matrix[time_inds, :]

plt.figure()
plt.plot(tim_crop, design_matrix_crop)

contrasts = np.array([(0, 1, -1, 0, 0),
                      (1, 0, -1, 0, 0)])
contrast_names = ['right_vs_rest',
                  'left_vs_rest']

glmdes = glmtools.design.GLMDesign.initialise_from_matrices(
    design_matrix_crop,
    contrasts,
    regressor_names=['left', 'right', 'rest', 'both', 'start_rest'],
    contrast_names=contrast_names)
#glmdes.plot_summary()

################
# fit GLM

glmdata = glmtools.data.ContinuousGLMData(data=parcel_timeseries['data'].T, sample_rate=raw.info['sfreq'])
model = glmtools.fit.OLSModel(glmdes, glmdata)
tstats = []
for cc in range(len(contrasts)):
    tstats.append(np.reshape(model.tstats[cc, :], [1, -1]))

################
# Write out parcellated stats as niftii vols and view them

con = 0
stats_dir = op.join(subjects_dir, subject, 'rhino', 'stats')
if not os.path.isdir(stats_dir):
    os.mkdir(stats_dir)
tstat_nii_fname = op.join(stats_dir, 'tstat{}_parcel.nii.gz'.format(cc + 1))

tstat_nii_fname = p.nii(tstats[con][0, :], method='assignments', out_nii_fname=tstat_nii_fname)
rhino.fsleyes_overlay(parcellation_background_fname, tstat_nii_fname)
