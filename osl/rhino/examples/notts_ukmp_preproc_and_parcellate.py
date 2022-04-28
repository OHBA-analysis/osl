#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:39:24 2021

Full pipeline (including preproc, source recon and parcellation) for getting parcel timeseries from Notts UKMP data

@author: woolrich

"""

import os
import os.path as op
from os import makedirs

import numpy as np
import pandas as pd
import yaml

from osl import rhino
from osl import report
from osl import parcellation
from osl import preprocessing

import mne

subjects_dir = '/Users/woolrich/homedir/vols_data/ukmp'

subjects_to_do = np.arange(1, 75)

# subjects_to_do = np.setdiff1d(subjects_to_do, subjects_to_exclude, assume_unique=True)
# subjects_to_do = {1, 2, 4, 5, 6, 7}
subjects_to_do = {1}

task = 'resteyesopen'
freq_range = (1, 45)
use_amplitude_timeseries = True
rank = {'mag': 125}
chantypes = ['mag']

subjects = []
ds_files = []
preproc_fif_files = []
smri_files = []
pos_files = []
recon_dirs = []

# input files
for sub in subjects_to_do:
    subject = 'sub-not00{}'.format(sub)
    subjects.append(subject)
    ds_files.append(op.join(subjects_dir, subject, 'meg', subject + '_task-' + task + '_meg.ds'))
    preproc_fif_files.append(op.join(subjects_dir, subject, 'meg', subject + '_task-' + task + '_meg_preproc_raw.fif'))
    pos_files.append(op.join(subjects_dir, subject, 'meg', subject + '_headshape.pos'))
    smri_files.append(op.join(subjects_dir, subject, 'anat', subject + '_T1w.nii.gz'))
    recon_dirs.append(op.join(subjects_dir, subject, 'meg'))

run_preproc = True
run_preproc_report = True
run_compute_surfaces = True
run_coreg = True
run_forward_model = True
run_source_recon_parcellate = True
run_orth = True
run_extract_parcel_timeseries = True

# parcellation to use
parcellation_fname = op.join('/Users/woolrich/Dropbox/vols_scripts/hmm_misc_funcs/parcellations',
                             'fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz')

std_brain = '/Users/woolrich/homedir/vols_data/mne/self_paced_fingertap/subject1/rhino/surfaces/MNI152_T1_brain_2mm.nii.gz'

# resolution of dipole grid for source recon
gridstep = 7  # mm

##########################

if run_preproc:

    config_text = """
    meta:
        event_codes:

    preproc:
        - resample:       {sfreq: 250, n_jobs: 6}            
        - filter:         {l_freq: 1, h_freq: 100}
        
    """

    # - bad_segments: {segment_len: 800, picks: 'meg'}
    # - bad_channels: {picks: 'meg'}
    config = yaml.load(config_text, Loader=yaml.FullLoader)

    # Process a single file, this outputs fif_file
    dataset = preprocessing.run_proc_batch(config, ds_files, outdir=subjects_dir, overwrite=True)

    for subject in subjects:
        os.system('mv {} {}'.format(
            op.join(subjects_dir, subject + '_task-' + task + '_meg_preproc_raw.*'),
            op.join(subjects_dir, subject, 'meg')
        ))

if run_preproc_report:
    report.gen_report(preproc_fif_files, outdir=subjects_dir)

##########################

if run_compute_surfaces:

    ###########
    # Compute surfaces
    for subject, smri_file in zip(subjects, smri_files):
        print('Compute surfaces for subject {}'.format(subject))

        rhino.compute_surfaces(smri_file,
                               subjects_dir, subject,
                               include_nose=True,
                               cleanup_files=True)

    if False:
        rhino.surfaces_display(subjects_dir, subjects[0])

##########################

if run_coreg:

    ###########
    # Setup polhemus points

    polhemus_nasion_files = []
    polhemus_rpa_files = []
    polhemus_lpa_files = []
    polhemus_headshape_files = []

    for pos_file, subject in zip(pos_files, subjects):
        # setup polhemus files
        polhemus_nasion_file = op.join(subjects_dir, subject, 'polhemus_nasion.txt')
        polhemus_rpa_file = op.join(subjects_dir, subject, 'polhemus_rpa.txt')
        polhemus_lpa_file = op.join(subjects_dir, subject, 'polhemus_lpa.txt')
        polhemus_headshape_file = op.join(subjects_dir, subject, 'polhemus_headshape.txt')

        # Load in txt file, these values are in cm in polhemus space:
        num_headshape_pnts = int(pd.read_csv(pos_file, header=None).to_numpy()[0])
        data = pd.read_csv(pos_file, header=None, skiprows=[0], delim_whitespace=True)

        # RHINO is going to work with distances in mm
        # So convert to mm from cm, note that these are in polhemus space
        data.iloc[:, 1:4] = data.iloc[:, 1:4] * 10

        # Polhemus fiducial points in polhemus space
        polhemus_nasion_polhemus = data[data.iloc[:, 0].str.match('nasion')].iloc[0, 1:4].to_numpy().astype('float64').T
        polhemus_rpa_polhemus = data[data.iloc[:, 0].str.match('right')].iloc[0, 1:4].to_numpy().astype('float64').T
        polhemus_lpa_polhemus = data[data.iloc[:, 0].str.match('left')].iloc[0, 1:4].to_numpy().astype('float64').T

        # Polhemus headshape points in polhemus space in mm
        polhemus_headshape_polhemus = data[0:num_headshape_pnts].iloc[:, 1:4].to_numpy().T

        np.savetxt(polhemus_nasion_file, polhemus_nasion_polhemus)
        np.savetxt(polhemus_rpa_file, polhemus_rpa_polhemus)
        np.savetxt(polhemus_lpa_file, polhemus_lpa_polhemus)
        np.savetxt(polhemus_headshape_file, polhemus_headshape_polhemus)

        polhemus_nasion_files.append(polhemus_nasion_file)
        polhemus_lpa_files.append(polhemus_lpa_file)
        polhemus_rpa_files.append(polhemus_rpa_file)
        polhemus_headshape_files.append(polhemus_headshape_file)

    for subject, preproc_fif_file, \
        polhemus_headshape_file, \
        polhemus_nasion_file, polhemus_rpa_file, polhemus_lpa_file in \
            zip(subjects, preproc_fif_files,
                polhemus_headshape_files,
                polhemus_nasion_files, polhemus_rpa_files, polhemus_lpa_files
                ):
        print('Coreg for subject {}'.format(subject))

        rhino.coreg(preproc_fif_file,
                subjects_dir, subject,
                polhemus_headshape_file,
                polhemus_nasion_file, polhemus_rpa_file, polhemus_lpa_file,
                use_headshape=True)

    # Purple dots are the polhemus derived fiducials
    # Yellow diamonds are the sMRI derived fiducials
    # Position of sMRI derived fiducials are the ones that are refined if
    # useheadshape=True was used for rhino.coreg

    if False:
        rhino.coreg_display(subjects_dir, subjects[0],
                        plot_type='surf',
                        display_outskin_with_nose=True,
                        display_sensors=True)

###########################

if run_forward_model:

    for subject in subjects:
        print('Forward model for subject {}'.format(subject))

        rhino.forward_model(subjects_dir, subject,
                        model='Single Layer',
                        gridstep=gridstep,
                        mindist=4.0)

    if False:
        rhino.bem_display(subjects_dir, subjects[0],
                      plot_type='surf',
                      display_outskin_with_nose=False,
                      display_sensors=True)

##########################

if run_source_recon_parcellate:

    def lcmv_beamformer(subjects_dir, subject, fif_file, freq_range):

        raw = mne.io.read_raw_fif(fif_file)

        # Use MEG sensors
        raw.pick(chantypes)

        raw.load_data()

        raw.filter(l_freq=freq_range[0], h_freq=freq_range[1], method='iir', iir_params={'order': 5, 'btype': 'bandpass', 'ftype': 'butter'})

        data_cov = mne.compute_raw_covariance(raw, method='empirical')

        fwd_fname = rhino.get_coreg_filenames(subjects_dir, subject)['forward_model_file']
        fwd = mne.read_forward_solution(fwd_fname)

        # make LCMV filter
        filters = rhino.make_lcmv(subjects_dir, subject,
                                  raw,
                                  chantypes,
                                  reg=0,
                                  pick_ori='max-power-pre-weight-norm',
                                  weight_norm='nai',
                                  rank=rank,
                                  reduce_rank=True,
                                  verbose=True)

        stc = mne.beamformer.apply_lcmv_raw(raw, filters)

        return stc

    parc = parcellation.Parcellation(parcellation_fname)

    for subject, preproc_fif_file, recon_dir in zip(subjects, preproc_fif_files, recon_dirs):

        print('Source recon and parcellation for subject {}'.format(subject))

        # load forward solution
        # stc is source space time series (in head/polhemus space)
        stc = lcmv_beamformer(subjects_dir, subject, preproc_fif_file, freq_range)

        # Convert stc from head/polhemus space to standard brain grid in MNI space
        recon_timeseries_mni, reference_brain_fname, recon_coords_mni, _ = rhino.transform_recon_timeseries(
            subjects_dir, subject,
            recon_timeseries=stc.data,
            reference_brain='mni')

        # Apply MNI space parcellation to voxelwise data (voxels x tpts) contained in recon_timeseries_mni
        # Resulting parc.parcel_timeseries will be (parcels x tpts)
        parc.parcellate(recon_timeseries_mni, recon_coords_mni, method='spatial_basis')

        #os.system('mkdir {}'.format(recon_dir))
        makedirs(recon_dir, exist_ok=True)
        parc.save_parcel_timeseries(op.join(recon_dir, 'parcel_timeseries.hd5'))

##########################

if run_orth:

    parc = parcellation.Parcellation(parcellation_fname)

    for recon_dir in recon_dirs:
        parc.load_parcel_timeseries(op.join(recon_dir, 'parcel_timeseries.hd5'))

        # parcel_timeseries['data'] is nparcels x ntpts
        parc.symmetric_orthogonalise(maintain_magnitudes=True)

        parc.save_parcel_timeseries(op.join(recon_dir, 'parcel_timeseries_orth.hd5'))

##########################

if run_extract_parcel_timeseries:
    print("Extract parcel timeseries")

    parc = parcellation.Parcellation(parcellation_fname)

    for recon_dir in recon_dirs:
        parc.load_parcel_timeseries(op.join(recon_dir, 'parcel_timeseries_orth.hd5'))

        if use_amplitude_timeseries:
            nparcels = parc.parcel_timeseries['data'].shape[0]
            hilb = np.zeros(parc.parcel_timeseries['data'].shape)
            for idx in range(nparcels):
                hilb[idx, :] = mne.filter._my_hilbert(parc.parcel_timeseries['data'][idx, :], None, True)

            np.save(op.join(recon_dir, 'parcel_timeseries_hilb.npy'), hilb.T.astype(np.float32))
        else:
            np.save(op.join(recon_dir, 'parcel_timeseries.npy'), parc.parcel_timeseries['data'].T.astype(np.float32))

if False:
    # copy to bmrc
    for subject, recon_dir in zip(subjects, recon_dirs):
        print('rsync -Phr {} vxw496@cluster1.bmrc.ox.ac.uk:{}'
              .format(op.join(recon_dir, 'parcel_timeseries_hilb.npy'),
                      op.join('/users/woolrich/vxw496/projects/notts_ukmp', subject, 'meg/')
                      )
              )


