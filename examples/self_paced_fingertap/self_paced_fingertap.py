#!/usr/bin/env python

"""
Note to self: See ~/CloudDocs/scripts/notts_ctf_fingertap.m for matlab equivalent
"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import glmtools
import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osl import rhino
import mne

import yaml
import osl

subjects_dir = "/Users/woolrich/homedir/vols_data/mne/self_paced_fingertap"
subject = "subject1"

# input files
ds_file = op.join(subjects_dir, subject, "JRH_MotorCon_20100429_01_FORMARK.ds")
fif_file = op.join(subjects_dir, subject, "JRH_MotorCon_20100429_01_FORMARK_raw.fif")
pos_file = op.join(subjects_dir, subject, "JH_Motorcon.pos")
smri_file = (
    "/Users/woolrich/homedir/vols_data/ukmp/sub-not002/anat/sub-not002_T1w.nii.gz"
)

run_sensorspace = False  # if false then do source space

run_preproc = False
run_compute_surfaces = True
run_coreg = True
run_forward_model = True

rank = {"mag": 125}
chantypes = ["mag"]

gridstep = 8  # mm

# -------------
# Preprocessing
if run_preproc:
    config = """
        preproc:
        - resample: {sfreq: 150, n_jobs: 6}    
        #- filter: {l_freq: 13, h_freq: 30}
    """

    # Process a single file, this outputs fif_file
    dataset = osl.preprocessing.run_proc_chain(
        ds_file, config, outdir=op.join(subjects_dir, subject), overwrite=True
    )

# ------------
# Get the data
raw = mne.io.read_raw_fif(fif_file)
raw.pick(chantypes)
raw.load_data()

# focus on beta band
raw.filter(
    l_freq=13,
    h_freq=30,
    method="iir",
    iir_params={"order": 5, "btype": "bandpass", "ftype": "butter"},
)

# -----------------------
# Establish design matrix

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
# in seconds is block_length:

# get time indices that correspond to the time window that was source
# reconstructed

# Use time window that excludes the initial rest period
time_from = 300  # secs
time_to = 1439.9933  # secs

ntotal_tpts = mne.io.read_raw_fif(fif_file).n_times
tres = 1 / raw.info["sfreq"]
timepnts = np.arange(0, tres * ntotal_tpts, step=tres)
time_inds = np.where((timepnts >= time_from) & (timepnts <= (time_to + tres)))[0]
times = timepnts[time_inds]

block_length = tres * int(30 / tres)  # secs
block_order = np.array(
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1, 2, 3, 1, 4, 3, 4, 1, 3, 2, 1,
     4, 4, 2, 1, 3, 3, 4, 1, 4, 3, 1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 3, 4, 1, 2]
)

# Create the design matrix:
design_matrix = np.zeros([ntotal_tpts, 5])

tim = 0
for tt in range(len(block_order)):
    design_matrix[tim : int(tim + block_length / tres), block_order[tt] - 1] = 1
    tim += int(block_length / tres)

tim_crop = timepnts[time_inds]
design_matrix_crop = design_matrix[time_inds, :]

plt.figure()
plt.plot(tim_crop, design_matrix_crop)
plt.show()

# Setup the GLM in glmtools
contrasts = np.array([(0, 1, -1, 0, 0), (1, 0, -1, 0, 0)])
contrast_names = ["right_vs_rest", "left_vs_rest"]

glmdes = glmtools.design.GLMDesign.initialise_from_matrices(
    design_matrix_crop,
    contrasts,
    regressor_names=["left", "right", "rest", "both", "start_rest"],
    contrast_names=contrast_names,
)

#glmdes.plot_summary()


def glm_fast(data, design_matrix, contrasts):
    pinvxtx = np.linalg.pinv(design_matrix.T @ design_matrix)
    pinvx = np.linalg.pinv(design_matrix)
    pe = pinvx @ data.T
    r = data.T - design_matrix @ pe
    vr = np.diag(r.T @ r / (data.shape[1] - design_matrix.shape[1]))
    cope = []
    tstat = []

    for cc in range(len(contrasts)):
        c = np.reshape(contrasts[cc], [-1, 1])
        varcope = (c.T @ pinvxtx @ c) * vr
        cope.append(c.T @ pe)
        tstat.append(cope[cc] / np.sqrt(varcope))
    return tstat, cope


# ----------------------
# Do GLM in sensor space

if run_sensorspace:

    # Do hilbert transform
    raw.apply_hilbert()
    raw.crop(time_from, time_to).load_data()

    data = np.abs(raw.get_data())

    if False:
        # Use glm_fast to do GLM
        tstats, copes = glm_fast(data, design_matrix_crop, contrasts)
    else:
        # Use glmtools to do GLM
        glmdata = glmtools.data.ContinuousGLMData(data=data.T, sample_rate=1 / tres)
        model = glmtools.fit.OLSModel(glmdes, glmdata)
        tstats = []
        for cc in range(len(contrasts)):
            tstats.append(np.reshape(model.tstats[cc, :], [1, -1]))

    # Display stats as sensor space topo plots

    fig, axs = plt.subplots(len(contrasts))

    for cc in range(len(contrasts)):
        im, cm = mne.viz.plot_topomap(
            tstats[cc][0, :], raw.info, axes=axs[cc], show=True, vmin=-80, vmax=0
        )

        axs[cc].set_title(contrast_names[cc])

        # manually fiddle the position of colorbar
        ax_x_start = 0.9
        ax_x_width = 0.02
        ax_y_start = 0.2
        ax_y_height = 0.5
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        clb.ax.set_title("tstat", fontsize=10)  # title on top of colorbar

else:

    # --------------------------------------------------------------------------
    # Get polhemus fids and headshape points into required file format for rhino
    # i.e. in polhemus space in mm

    # setup filenames
    polhemus_nasion_file = op.join(subjects_dir, subject, "polhemus_nasion.txt")
    polhemus_rpa_file = op.join(subjects_dir, subject, "polhemus_rpa.txt")
    polhemus_lpa_file = op.join(subjects_dir, subject, "polhemus_lpa.txt")
    polhemus_headshape_file = op.join(subjects_dir, subject, "polhemus_headshape.txt")

    # Load in txt file, these values are in cm in polhemus space:
    num_headshape_pnts = int(pd.read_csv(pos_file, header=None).to_numpy()[0])
    data = pd.read_csv(pos_file, header=None, skiprows=[0], delim_whitespace=True)

    # RHINO is going to work with distances in mm
    # So convert to mm from cm, note that these are in polhemus space
    data.iloc[:, 1:4] = data.iloc[:, 1:4] * 10

    # Polhemus fiducial points in polhemus space
    polhemus_nasion_polhemus = (
        data[data.iloc[:, 0].str.match("nasion")]
        .iloc[0, 1:4]
        .to_numpy()
        .astype("float64")
        .T
    )
    polhemus_rpa_polhemus = (
        data[data.iloc[:, 0].str.match("right")]
        .iloc[0, 1:4]
        .to_numpy()
        .astype("float64")
        .T
    )
    polhemus_lpa_polhemus = (
        data[data.iloc[:, 0].str.match("left")]
        .iloc[0, 1:4]
        .to_numpy()
        .astype("float64")
        .T
    )

    # Polhemus headshape points in polhemus space in mm
    polhemus_headshape_polhemus = data[0:num_headshape_pnts].iloc[:, 1:4].to_numpy().T

    np.savetxt(polhemus_nasion_file, polhemus_nasion_polhemus)
    np.savetxt(polhemus_rpa_file, polhemus_rpa_polhemus)
    np.savetxt(polhemus_lpa_file, polhemus_lpa_polhemus)
    np.savetxt(polhemus_headshape_file, polhemus_headshape_polhemus)

    # --------------------------------------
    # Compute surfaces, coreg, forward model
    if run_compute_surfaces:
        rhino.compute_surfaces(
            smri_file, subjects_dir, subject, include_nose=True, cleanup_files=True
        )

        rhino.surfaces_display(subjects_dir, subject)

    if run_coreg:
        # call rhino
        rhino.coreg(
            fif_file,
            subjects_dir,
            subject,
            polhemus_headshape_file,
            polhemus_nasion_file,
            polhemus_rpa_file,
            polhemus_lpa_file,
            use_headshape=True,
        )

        # Purple dots are the polhemus derived fiducials
        # Yellow diamonds are the sMRI derived fiducials
        # Position of sMRI derived fiducials are the ones that are refined if
        # useheadshape=True was used for rhino.coreg
        rhino.coreg_display(
            subjects_dir,
            subject,
            plot_type="surf",
            display_outskin_with_nose=True,
            display_sensors=True,
        )

    #  Forward modelling
    if run_forward_model:
        rhino.forward_model(
            subjects_dir, subject, model="Single Layer", gridstep=gridstep, mindist=4.0
        )

        rhino.bem_display(
            subjects_dir,
            subject,
            plot_type="surf",
            display_outskin_with_nose=False,
            display_sensors=True,
        )

    # -------------------------
    # Take a look at leadfields

    # We can explore the content of fwd to access the numpy array that contains
    # the gain matrix

    # load forward solution
    fwd_fname = rhino.get_coreg_filenames(subjects_dir, subject)["forward_model_file"]
    fwd = mne.read_forward_solution(fwd_fname)

    leadfield = fwd["sol"]["data"]
    print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

    # ------------
    # Source recon

    # make LCMV filter
    filters = rhino.make_lcmv(
        subjects_dir,
        subject,
        raw,
        chantypes,
        reg=0,
        pick_ori="max-power-pre-weight-norm",
        weight_norm="nai",
        rank=rank,
        reduce_rank=True,
        verbose=True,
    )

    # plot data covariance matrix
    filters["data_cov"].plot(raw.info)

    # plot noise covariance the inverse of which is used for whitening
    plt.figure()
    plt.plot(filters["noise_cov"].data)
    plt.title("noise cov")

    # Do hilbert transform
    hilb_raw = raw.copy()
    hilb_raw.apply_hilbert()

    # stc is source space time series (in head/polhemus space)
    stc = mne.beamformer.apply_lcmv_raw(hilb_raw, filters)

    if False:
        # debugging code

        # use in conjuction with run_wakeman_henson, and
        # these line inserted at line 61 in spm-bemforming-toolbox/bf_inverse_lcmv_multicov.m
        #   tmp = readNPY('/Users/woolrich/homedir/vols_data/mne/self_paced_fingertap/subject1/leadfield2save.npy');
        #   S.L{1692} = tmp;

        chanind = 97
        raw.info["ch_names"][chanind]
        plt.figure()
        plt.plot(raw.get_data()[chanind, 999:1200])
        plt.plot(np.abs(hilb_raw.get_data()[97, 999:1200]))
        plt.title("sensor {}".format(raw.info["ch_names"][chanind]))

        stc = mne.beamformer.apply_lcmv_raw(raw, filters, max_ori_out="signed")
        (
            recon_timeseries_mni,
            reference_brain_fname,
            recon_coords_mni,
            recon_indices,
        ) = rhino.transform_recon_timeseries(
            subjects_dir, subject, recon_timeseries=stc.data, reference_brain="mni"
        )
        vox = rhino.utils._closest_node((-40, -25, 56), recon_coords_mni.T)[0]

        plt.figure()
        print(recon_coords_mni[:, vox])  # 2541
        plt.plot(recon_timeseries_mni[vox, 999:1200])
        print(int(recon_indices[vox]))
        plt.title("vox {}".format(vox))

        leadfield = fwd["sol"]["data"]
        plt.figure()
        plt.plot(np.reshape(leadfield, [273, -1, 3])[:, int(recon_indices[vox]), 0])
        plt.plot(np.reshape(leadfield, [273, -1, 3])[:, int(recon_indices[vox]), 1])
        plt.plot(np.reshape(leadfield, [273, -1, 3])[:, int(recon_indices[vox]), 2])

        leadfield2save = np.zeros([leadfield.shape[0], 3])
        for ii in range(3):
            leadfield2save[:, ii] = np.reshape(leadfield, [273, -1, 3])[
                :, int(recon_indices[vox]), ii
            ]
        np.save(op.join(subjects_dir, subject, "leadfield2save"), leadfield2save)

        np.save(op.join(subjects_dir, subject, "data_cov"), data_cov.data)

    # ------------------------------------------------------------
    # Fit GLM to hilbert envelope in source space contained in stc

    # hilbert transform gave us complex data, we want the amplitude
    data = np.abs(stc.data)

    data = data[:, time_inds]

    if False:
        tstats, copes = glm_fast(data, design_matrix_crop, contrasts)
    else:
        glmdata = glmtools.data.ContinuousGLMData(
            data=data.T, sample_rate=1 / stc.tstep
        )
        model = glmtools.fit.OLSModel(glmdes, glmdata)
        tstats = []
        for cc in range(len(contrasts)):
            tstats.append(np.reshape(model.tstats[cc, :], [1, -1]))

    # Write out stats as niftii vols
    stats_dir = op.join(subjects_dir, subject, "rhino", "stats")
    if not os.path.isdir(stats_dir):
        os.mkdir(stats_dir)

    # setup filenames and stats to write out
    volumes = []
    nii_file_names = []
    for cc in range(len(contrasts)):
        volumes.append(tstats[cc][0, :].T)
        nii_file_names.append(op.join(stats_dir, "tstat{}.nii.gz".format(cc + 1)))

    # Write stats as niftii file on a standard brain grid in MNI space
    con = 0
    out_nii_fname, stdbrain_mask_fname = rhino.recon_timeseries2niftii(
        subjects_dir,
        subject,
        recon_timeseries=volumes[con],
        out_nii_fname=nii_file_names[con],
        reference_brain="mni",
        times=raw.times,
    )

    rhino.fsleyes_overlay(stdbrain_mask_fname, out_nii_fname)
