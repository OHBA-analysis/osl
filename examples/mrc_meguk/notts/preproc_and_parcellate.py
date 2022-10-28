#!/usr/bin/env python

"""Full pipeline for getting parcel time series from Nottingham MEG UK partnership data.

Includes preprocessing, beamforming and parcellation.
"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import os.path as op

import numpy as np
import pandas as pd

from osl import preprocessing
from osl import source_recon

subjects_dir = "/Users/woolrich/homedir/vols_data/ukmp"
subjects_to_do = np.arange(75)+1
subjects_to_do = np.arange(2)+1

task = "resteyesopen"

subjects = []
ds_files = []
preproc_fif_files = []
smri_files = []

recon_dir = op.join(subjects_dir, 'recon')
POS_FILE = subjects_dir + "/{0}/meg/{0}_headshape.pos"

# input files
for sub in subjects_to_do:
    subject = "sub-not" + ("{}".format(sub)).zfill(3)
    ds_file = op.join(
        subjects_dir, subject, "meg", subject + "_task-" + task + "_meg.ds"
    )
    smri_file = op.join(subjects_dir, subject, "anat", subject + "_T1w.nii.gz")

    # check ds file and structural file exists for this subject
    if op.exists(ds_file) and op.exists(smri_file):
        ds_files.append(ds_file)
        subjects.append(subject)
        preproc_fif_files.append(
            op.join(
                subjects_dir,
                subject,
                "meg",
                subject + "_task-" + task + "_meg_preproc_raw.fif",
            )
        )

        smri_files.append(smri_file)

run_preproc = True
run_beamform_and_parcellate = False
run_fix_sign_ambiguity = False

# parcellation to use
parcellation_fname = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

# resolution of dipole grid for source recon
gridstep = 7  # mm

if run_preproc:

    # Note that with CTF mne.pick_types will return:
    # ~274 axial grads (as magnetometers) if {picks: 'mag', ref_meg: False}
    # ~28 reference axial grads if {picks: 'grad'}

    config = """
     preproc:
        - resample:     {sfreq: 250}
        - filter:       {l_freq: 0.5, h_freq: 100, method: 'iir', iir_params: {order: 5, ftype: butter}}
        - crop:         {tmin: 10, tmax: 300}           
        - set_channel_types: {EEG057: eog, EEG058: eog, EEG059: ecg}
        - bad_channels: {picks: 'mag', ref_meg: False, significance_level: 0.1}
        - bad_channels: {picks: 'grad', significance_level: 0.4}
        - bad_segments: {segment_len: 600, picks: 'mag', ref_meg: False, significance_level: 0.1}
    """

    #         - crop:         {tmin: 20, tmax: 280}
    #- bad_channels: {picks: 'meg', significance_level: 0.4}
    #- bad_segments: {segment_len: 600, picks: 'meg', significance_level: 0.1}

    # Process a single file, this outputs fif_file
    dataset = preprocessing.run_proc_batch(
        config, ds_files, outdir=subjects_dir, overwrite=True
    )

    for subject in subjects:
        os.system(
            "mv {} {}".format(
                op.join(subjects_dir, subject + "_task-" + task + "_meg_preproc_raw.*"),
                op.join(subjects_dir, subject, "meg"),
            )
        )

# -------------------------------------------------------------
# %% Coreg and Source recon and Parcellate

def save_polhemus_from_pos(src_dir, subject, preproc_file, smri_file, logger):
    """Saves fiducials/headshape from a pos file."""

    # Load pos file
    pos_file = POS_FILE.format(subject)
    logger.info(f"Saving polhemus from {pos_file}")

    #  Get coreg filenames
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load in txt file, these values are in cm in polhemus space:
    num_headshape_pnts = int(pd.read_csv(pos_file, header=None).to_numpy()[0])
    data = pd.read_csv(pos_file, header=None, skiprows=[0], delim_whitespace=True)

    # RHINO is going to work with distances in mm
    # So convert to mm from cm, note that these are in polhemus space
    data.iloc[:, 1:4] = data.iloc[:, 1:4] * 10

    # Polhemus fiducial points in polhemus space
    polhemus_nasion = (
        data[data.iloc[:, 0].str.match("nasion")]
            .iloc[0, 1:4].to_numpy().astype("float64").T
    )
    polhemus_rpa = (
        data[data.iloc[:, 0].str.match("right")]
            .iloc[0, 1:4].to_numpy().astype("float64").T
    )
    polhemus_lpa = (
        data[data.iloc[:, 0].str.match("left")]
            .iloc[0, 1:4].to_numpy().astype("float64").T
    )

    # Polhemus headshape points in polhemus space in mm
    polhemus_headshape = (
        data[0:num_headshape_pnts]
            .iloc[:, 1:4].to_numpy().astype("float64").T
    )

    # Save
    np.savetxt(filenames["polhemus_nasion_file"], polhemus_nasion)
    np.savetxt(filenames["polhemus_rpa_file"], polhemus_rpa)
    np.savetxt(filenames["polhemus_lpa_file"], polhemus_lpa)
    np.savetxt(filenames["polhemus_headshape_file"], polhemus_headshape)

if run_beamform_and_parcellate:

    # Settings
    config = """
        source_recon:
        - save_polhemus_from_pos: {}
        - coregister:
            include_nose: true
            use_nose: true
            use_headshape: true
            model: Single Layer
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: mag
            rank: {mag: 120}
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
        extra_funcs=[save_polhemus_from_pos],
    )


if False:

    # to just run surfaces for subject 0:
    source_recon.rhino.compute_surfaces(
        smri_files[0],
        recon_dir,
        subjects[0],
        overwrite=True
    )

    # to view surfaces for subject 0:
    source_recon.rhino.surfaces_display(recon_dir, subjects[3])

    # to just run coreg for subject 0:
    source_recon.rhino.coreg(
        preproc_fif_files[0],
        recon_dir,
        subjects[0],
        already_coregistered=True
    )

    # to view coreg result for subject 0:
    source_recon.rhino.coreg_display(recon_dir, subjects[0],
                                     plot_type='surf')

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
            n_embeddings: 13
            standardize: True
            n_init: 3
            n_iter: 2500
            max_flips: 20
    """

    # Do the sign flipping
    source_recon.run_src_batch(config, recon_dir, subjects, report_name='sflip_report')

    if True:
        # copy sf files to a single directory (makes it easier to copy minimal files to, e.g. BMRC, for downstream analysis)
        os.makedirs(op.join(recon_dir, 'sflip_data'), exist_ok=True)

        sflip_parc_files = []
        for subject in subjects:
            sflip_parc_file_from = op.join(recon_dir, subject, 'sflip_parc.npy')
            sflip_parc_file_to = op.join(recon_dir, 'sflip_data', subject + '_sflip_parc.npy')

            os.system('cp -f {} {}'.format(sflip_parc_file_from, sflip_parc_file_to))

            sflip_parc_files.append(sflip_parc_file_to)




