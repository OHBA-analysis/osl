#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""epoch
Created on Fri Oct 15 16:40:02 2021

Run parcellation based GLM analysis on Wakeman-Henson dataset

@author: woolrich
"""

import os
import os.path as op

import numpy as np
import matplotlib.pyplot as plt
from osl import rhino
from osl import parcellation
import osl
import mne
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import glmtools as glm

base_dir = '/Users/woolrich/homedir/vols_data/WakeHen'
subjects_dir = op.join(base_dir, 'wakehen_glm')
subject = 'subject1_run2'

# load precomputed forward solution
fwd_fname = rhino.get_coreg_filenames(subjects_dir, subject)['forward_model_file']
fwd = mne.read_forward_solution(fwd_fname)

# fif file
fif_file_in = op.join(base_dir, 'raw/sub001/MEG/run_02_sss.fif')

# parcellation to use
parcellation_fname = op.join('/Users/woolrich/Dropbox/vols_scripts/hmm_misc_funcs/parcellations',
                             'fmri_d100_parcellation_with_PCC_reduced_2mm.nii.gz')

# nii volume to overlay parcellation results on
parcellation_background_fname = op.join(os.environ['FSLDIR'], 'data/standard/MNI152_T1_2mm_brain.nii.gz')

run_forward_model = True
run_recon = True
baseline_correct = True

gridstep = 8  # mm

################

use_eeg = False
use_meg = True

chantypes = []
chantypes_reject_thresh = {'eog': 250e-6}
rank= {}

if use_eeg:
    chantypes.append('eeg')
    rank.update({'eeg':30})
    chantypes_reject_thresh.update({'eeg': 250e-6})

if use_meg:
    chantypes.append('mag'),
    chantypes.append('grad'),
    rank.update({'meg': 60})
    chantypes_reject_thresh.update({'mag': 4e-12, 'grad': 4000e-13})

print('Channel types to use: {}'.format(chantypes))
print('Channel types and ranks for source recon: {}'.format(rank))

# -------------------------------------------------------------
# %% Preproc

# Load config file
base_dir = '/Users/woolrich/homedir/vols_data/WakeHen/'
config = osl.preprocessing.load_config(op.join(base_dir, 'wakehen_preproc.yml'))

# Run preproc
dataset = osl.preprocessing.run_proc_chain(fif_file_in, config, outdir=base_dir, overwrite=True)

dataset['raw'].filter(l_freq=1, h_freq=30, method='iir',
                      iir_params={'order': 5, 'btype': 'bandpass', 'ftype': 'butter'})

epochs = mne.Epochs(dataset['raw'],
                    dataset['events'],
                    dataset['event_id'],
                    tmin=-0.5, tmax=1.5,
                    baseline=(None, 0))

epochs.drop_bad(reject=chantypes_reject_thresh, verbose=True)

epochs.load_data()

if use_eeg:
    epochs.set_eeg_reference(ref_channels='average', projection=True)

epochs.pick(chantypes)

#############################################################################

if run_forward_model:
    if use_eeg:
        model = 'Triple Layer'
    else:
        model = 'Single Layer'

    rhino.forward_model(subjects_dir, subject,
                        model=model,
                        eeg=use_eeg,
                        meg=use_meg,
                        gridstep=gridstep,
                        mindist=4.0)

# -------------------------------------------------------------
# %% Source Recon

recon_dir = op.join(subjects_dir, subject, 'rhino', 'recon')
if not os.path.isdir(recon_dir):
    os.mkdir(recon_dir)

if run_recon:
    # make LCMV filter
    filters = rhino.make_lcmv(subjects_dir, subject,
                              epochs,
                              chantypes,
                              reg=0,
                              pick_ori='max-power-pre-weight-norm',
                              weight_norm='nai',
                              rank=rank,
                              reduce_rank=True,
                              verbose=True)

    # plot data covariance matrix
    filters['data_cov'].plot(epochs.info)

    # Apply filters to epoched data
    # stc is list of source space trial time series (in head/polhemus space)
    stc = apply_lcmv_epochs(epochs, filters, max_ori_out='signed')

    # Turn stc into a  nsources x ntpts x ntrials array
    sourcespace_epoched_data = []
    for trial in stc:
        sourcespace_epoched_data.append(trial.data)
    sourcespace_epoched_data = np.stack(sourcespace_epoched_data)
    sourcespace_epoched_data = np.transpose(sourcespace_epoched_data, [1, 2, 0])

    # Convert to standard brain grid in MNI space (to pass to parcellation)
    # (sourcespace_epoched_data is in head/polhemus space)
    recon_timeseries_mni, reference_brain_fname, recon_coords_mni, _ = rhino.transform_recon_timeseries(
        subjects_dir, subject,
        recon_timeseries=sourcespace_epoched_data,
        reference_brain='mni')

    np.save(op.join(recon_dir, 'recon_timeseries_mni.npy'), recon_timeseries_mni)
    np.save(op.join(recon_dir, 'recon_coords_mni.npy'), recon_coords_mni)

# -------------------------------------------------------------
# %% Parcellate

recon_timeseries_mni = np.load(op.join(recon_dir, 'recon_timeseries_mni.npy'))
recon_coords_mni = np.load(op.join(recon_dir, 'recon_coords_mni.npy'))

# plot centre of mass for each parcel
p = parcellation.Parcellation(parcellation_fname)
p.plot()

# Apply parcellation to voxelwise data (voxels x tpts x trials) contained in recon_timeseries_mni
# Resulting parcel_timeseries will be (parcels x tpts x trials)
parcel_timeseries = p.parcellate(recon_timeseries_mni, recon_coords_mni, method='spatial_basis')

# -------------------------------------------------------------
# %% Setup design matrix

DC = glm.design.DesignConfig()
DC.add_regressor(name='FamousFirst', rtype='Categorical', codes=5)
DC.add_regressor(name='FamousImmediate', rtype='Categorical', codes=6)
DC.add_regressor(name='FamousLast', rtype='Categorical', codes=7)
DC.add_regressor(name='UnfamiliarFirst', rtype='Categorical', codes=13)
DC.add_regressor(name='UnfamiliarImmediate', rtype='Categorical', codes=14)
DC.add_regressor(name='UnfamiliarLast', rtype='Categorical', codes=15)
DC.add_regressor(name='ScrambledFirst', rtype='Categorical', codes=17)
DC.add_regressor(name='ScrambledImmediate', rtype='Categorical', codes=18)
DC.add_regressor(name='ScrambledLast', rtype='Categorical', codes=19)
DC.add_simple_contrasts()
DC.add_contrast(name='Famous', values={'FamousFirst': 1, 'FamousImmediate': 1, 'FamousLast': 1})
DC.add_contrast(name='Unfamiliar', values={'UnfamiliarFirst': 1, 'UnfamiliarImmediate': 1, 'UnfamiliarLast': 1})
DC.add_contrast(name='Scrambled', values={'ScrambledFirst': 1, 'ScrambledImmediate': 1, 'ScrambledLast': 1})
DC.add_contrast(name='FamScram', values={'FamousFirst': 1, 'FamousLast': 1, 'ScrambledFirst': -1, 'ScrambledLast': -1})
DC.add_contrast(name='FirstLast', values={'FamousFirst': 1, 'FamousLast': -1, 'ScrambledFirst': 1, 'ScrambledLast': 1})
DC.add_contrast(name='Interaction',
                values={'FamousFirst': 1, 'FamousLast': -1, 'ScrambledFirst': -1, 'ScrambledLast': 1})
DC.add_contrast(name='Visual', values={'FamousFirst': 1, 'FamousImmediate': 1, 'FamousLast': 1, 'UnfamiliarFirst': 1,
                                       'UnfamiliarImmediate': 1, 'UnfamiliarLast': 1, 'ScrambledFirst': 1,
                                       'ScrambledImmediate': 1, 'ScrambledLast': 1})

print(DC.to_yaml())

data = glm.io.load_mne_epochs(epochs)

# Create Design Matrix
des = DC.design_from_datainfo(data.info)
des.plot_summary(show=True, savepath=fif_file_in.replace('.fif', '_design.png'))

# -------------------------------------------------------------
# %% Fit GLM to parcellated source recon data

# Create GLM data

# this is nsources x ntpts x ntrials
parcel_timeseries_data = parcel_timeseries['data']
# glm.data.TrialGLMData needs it to be ntrials x nsources x ntpts
data = glm.data.TrialGLMData(data=np.transpose(parcel_timeseries_data, (2, 0, 1)))
# Fit Model
model_source = glm.fit.OLSModel(des, data)

# ------------------------------
# %% Compute stats of interest from GLM fit

contrast_of_interest = 15

# take abs(cope) due to 180 degree ambiguity in dipole orientation
acope = np.abs(model_source.copes[contrast_of_interest])

# globally normalise by the mean
acope = acope / np.mean(acope)

if baseline_correct:
    baseline_mean = np.mean(abs(model_source.copes[contrast_of_interest][:, epochs.times < 0]), 1)
    # acope = acope - np.reshape(baseline_mean.T,[acope.shape[0],1])
    acope = acope - np.reshape(baseline_mean, [-1, 1])

# ------------------------------------------------------
# %% Output stats as 3D nii files at tpt of interest

stats_dir = op.join(subjects_dir, subject, 'rhino', 'stats')
if not os.path.isdir(stats_dir):
    os.mkdir(stats_dir)

# output nii nearest to this time point in msecs:
tpt = 0.110 + 0.034
volume_num = epochs.time_as_index(tpt)[0]

acope_nii_fname = p.nii(acope[:, volume_num], method='assignments')
rhino.fsleyes_overlay(parcellation_background_fname, acope_nii_fname)

# ------------------------------------------------------
# %% plot cope time course from an example parcel

parcel_index = 10
plt.figure()
plt.plot(epochs.times, acope[parcel_index,:])
plt.title('abs(cope) for contrast {}, for parcel {}'.format(
    contrast_of_interest, parcel_index))
plt.xlabel('time (s)')
plt.ylabel('abs(cope)')

# ------------------------------------------------------
# %% Write cope as 4D niftii file on a standard brain grid in MNI space
# 4th dimension is timepoint within a trial

out_nii_fname = op.join(stats_dir, 'acope{}_mni_parcel.nii.gz'.format(contrast_of_interest, gridstep))
acope_nii_fname = p.nii(acope, method='assignments',
                        times=epochs.times,
                        out_nii_fname=out_nii_fname)
rhino.fsleyes_overlay(parcellation_background_fname, acope_nii_fname)

# From fsleyes drop down menus Select "View/Time series"
# To see time labelling in secs:
#   - In the Time series panel, select Settings (the spanner icon)
#   - In the Time series settings popup, select "Use Pix Dims"