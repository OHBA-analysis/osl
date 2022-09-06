#!/usr/bin/env python

"""
Usage: For Human evoked field MEG data.

Links for paper and data/scripts:
- https://www.sciencedirect.com/science/article/pii/S1053811920302846?via%3Dihub
- https://zenodo.org/record/3233557
        
Note: The code is used in the study:
- 'Comparison of beamformer implementations for MEG source localizations'
"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>

from scipy.io import loadmat
import os
import os.path as op

from collections import OrderedDict
import mne
import numpy as np
import matplotlib.pyplot as plt
from osl import rhino

from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
import warnings

# if do_rhino is false then does standard MNE using precomputed coreg based on
# freesurfer surfaces
do_rhino = True

warnings.simplefilter("ignore", category=DeprecationWarning)
print(__doc__)

data_dir = "/Users/woolrich/homedir/vols_data/BeamComp_DataRepo/"
data_path = data_dir + "MEG/Human_EF/"
subjects_dir, subject = data_dir + "MRI/", "BeamCompMRI"


def my_var_cut_fn(epochs, plow, phigh, to_plot=True):
    """
    Variance base trial rejection function
    """
    trl_var, trlindx = np.empty((0, 1), "float"), np.arange(0, len(epochs))
    for trnum in range(len(epochs)):
        trl_var = np.vstack(
            (trl_var, max(np.var(np.squeeze(epochs[trnum].get_data()), axis=1)))
        )
    lim1 = (trl_var < np.percentile(trl_var, plow, interpolation="midpoint")).flatten()
    lim2 = (trl_var > np.percentile(trl_var, phigh, interpolation="midpoint")).flatten()
    outlr_idx = trlindx[lim1].tolist() + trlindx[lim2].tolist()
    if to_plot:
        plt.figure(), plt.scatter(
            trlindx, trl_var, marker="o", s=50, c="g", label="Good trials"
        ),
        plt.ylabel("Max. Variance accros channels-->")
        plt.scatter(
            outlr_idx,
            trl_var[outlr_idx],
            marker="o",
            s=50,
            c="r",
            label="Variance based bad trails",
        ),
        plt.xlabel("Trial number-->")
        plt.scatter(
            badtrls,
            trl_var[badtrls],
            marker="o",
            s=50,
            c="orange",
            label="Manually assigned bad trials",
        )
        plt.ylim(
            min(trl_var) - min(trl_var) * 0.01, max(trl_var) + max(trl_var) * 0.01
        ), plt.title(" Max. variance distribution")
        plt.legend()
        plt.show()
    bad_trials = np.union1d(badtrls, outlr_idx)
    print("Removed trials: %s\n" % bad_trials)
    return bad_trials


# %% Set parameters, directories and file names
more_plots = False
par = {"badch": [], "stimch": "STI 014"}
code_dir = "/Users/woolrich/CloudDocs/scripts/LCMV_pipelines/"

par["event_dict"] = OrderedDict()
par["event_dict"]["VEF_UR"] = 1
par["event_dict"]["VEF_LR"] = 2
par["event_dict"]["AEF_Re"] = 3
# par['event_dict']['VEF_LL']=4
# par['event_dict']['AEF_Le']=5
# par['event_dict']['VEF_UL']=8
# par['event_dict']['SEF_Lh']=16
# par['event_dict']['SEF_Rh']=32
st_len = len(par["event_dict"])

act_dip_ = loadmat(code_dir + "multimodal_biomag_Xfit_results.mat")
act_dip = act_dip_["multimodal_biomag_Xfit_diploc"][:, 3:6]

par["act_loc"] = OrderedDict()
par["act_loc"]["VEF_UR"] = act_dip[0]
par["act_loc"]["VEF_LR"] = act_dip[1]
par["act_loc"]["AEF_Re"] = act_dip[2]
# par['act_loc']['VEF_LL']=act_dip[3]
# par['act_loc']['AEF_Le']=act_dip[4]
# par['act_loc']['VEF_UL']=act_dip[5]
# par['act_loc']['SEF_Lh']=act_dip[6]
# par['act_loc']['SEF_Rh']=act_dip[7]

filename = "multimodal_raw.fif"  # or, multimodal_raw_tsss.fif
fname = data_path + filename
# trans = subjects_dir + subject + '/mri/transforms/' + subject + '-trans.fif'
trans = (
    subjects_dir
    + subject
    + "//mri/brain-neuromag/sets//"
    + "BeamCompMRI-amit-131118-MNEicp-trans.fif"
)
mrifile = subjects_dir + subject + "/mri/T1.mgz"
surffile = subjects_dir + subject + "/bem/watershed/" + subject + "_brain_surface"
dfname = os.path.split(os.path.splitext(fname)[0])[1]

# %% Load data
raw = mne.io.read_raw_fif(fname, allow_maxshield=False, preload=True, verbose=True)
if not raw.info["projs"] == []:
    bads = ["MEG 0442"]
    raw.drop_channels(bads)
events = mne.find_events(
    raw, stim_channel=par["stimch"], min_duration=0.001, shortest_event=1
)
events = events[events[:, 1] == 0]
if more_plots:
    mne.viz.plot_events(
        events, first_samp=0, event_id=par["event_dict"], equal_spacing=True, show=True
    )
raw.pick_types(meg=True)

# %% Apply filter if required
raw.filter(
    2,
    95,
    picks=None,
    filter_length="auto",
    l_trans_bandwidth="auto",
    h_trans_bandwidth="auto",
    n_jobs=1,
    method="fir",
    iir_params=None,
    phase="zero",
    fir_window="hamming",
    fir_design="firwin",
    skip_by_annotation=("edge", "bad_acq_skip"),
    pad="reflect_limited",
    verbose=True,
)

if more_plots:
    raw.plot(events=events)
    raw.plot_psd(fmin=0, fmax=100, proj=False, verbose=True)

# %% Epoching the data
reject = dict(grad=7000e-13, mag=7e-12, eog=250e-6)
epochs = mne.Epochs(
    raw,
    events,
    par["event_dict"],
    -0.5,
    0.5,
    baseline=(-0.5, 0),
    picks=None,
    preload=True,
    reject=None,
    flat=None,
    proj=False,
    decim=1,
    reject_tmin=None,
    reject_tmax=None,
    detrend=None,
    on_missing="error",
    reject_by_annotation=True,
    verbose=True,
)

if more_plots:
    for stimcat in list(par["event_dict"].keys())[0:8]:
        epochs[stimcat].average().plot(
            spatial_colors=True, titles=stimcat, gfp=True, time_unit="ms"
        )

# Compute Source Space && forward solution/leadfield
if do_rhino:

    run_compute_surfaces = True
    run_coreg = True
    run_forward_model = True

    # input files
    smri_file = op.join(subjects_dir, "BeamCompMRI/mri/T1.nii.gz")

    fif_file_preproc = op.join(data_path, filename)

    gridstep = 8  # mm

    # Setup polhemus files for coreg
    outdir = op.join(subjects_dir, subject)
    (
        polhemus_headshape_file,
        polhemus_nasion_file,
        polhemus_rpa_file,
        polhemus_lpa_file,
    ) = rhino.extract_polhemus_from_info(fif_file_preproc, outdir)

    # delete nose headshape pnts
    pnts = np.loadtxt(polhemus_headshape_file)

    if run_compute_surfaces:
        rhino.compute_surfaces(
            smri_file, subjects_dir, subject, include_nose=True, cleanup_files=True
        )

        rhino.surfaces_display(subjects_dir, subject)

    if run_coreg:
        # call rhino
        rhino.coreg(
            fif_file_preproc,
            subjects_dir,
            subject,
            polhemus_headshape_file,
            polhemus_nasion_file,
            polhemus_rpa_file,
            polhemus_lpa_file,
            use_nose=True,
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

    fwd_fname = rhino.get_coreg_filenames(subjects_dir, subject)["forward_model_file"]
    fwd = mne.read_forward_solution(fwd_fname)

else:
    # copy BEM into bem directory from backup
    os.system(
        "cp -rf /Users/woolrich/homedir/vols_data/BeamComp_DataRepo/MRI/BeamCompMRI/bem_backup/* /Users/woolrich/homedir/vols_data/BeamComp_DataRepo/MRI/BeamCompMRI/bem/"
    )
    model = mne.make_bem_model(
        subject=subject,
        ico=4,
        conductivity=(0.33,),
        subjects_dir=subjects_dir,
        verbose=True,
    )
    bem = mne.make_bem_solution(model)

    src_vol = mne.setup_volume_source_space(
        subject=subject,
        pos=5.0,
        mri=mrifile,
        bem=None,
        surface=surffile,
        mindist=2.5,
        exclude=10.0,
        subjects_dir=subjects_dir,
        volume_label=None,
        add_interpolator=True,
        verbose=True,
    )

    if more_plots:
        mne.viz.plot_bem(
            subject=subject,
            subjects_dir=subjects_dir,
            orientation="coronal",
            slices=range(73, 193, 5),
            brain_surfaces="pial",
            src=src_vol,
            show=True,
        )
        mne.viz.plot_alignment(
            epochs.info,
            trans=trans,
            subject=subject,
            subjects_dir=subjects_dir,
            fig=None,
            surfaces=["head-dense", "inner_skull"],
            coord_frame="head",
            show_axes=True,
            meg=True,
            eeg="original",
            dig=True,
            ecog=True,
            bem=None,
            seeg=True,
            src=src_vol,
            mri_fiducials=False,
            verbose=True,
        )

    fwd = mne.make_forward_solution(
        epochs.info,
        trans=trans,
        src=src_vol,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=2.5,
        n_jobs=4,
    )

print("Leadfield size : %d sensors x %d dipoles" % fwd["sol"]["data"].shape)

# %% Apply LCMV Beamformer
for stimcat in list(par["event_dict"].keys()):
    print(stimcat)
    if "VEF" in stimcat:
        par["ctrlwin"] = [-0.200, -0.050]
        par["actiwin"] = [0.050, 0.200]
    elif "AEF" in stimcat:
        par["ctrlwin"] = [-0.150, -0.020]
        par["actiwin"] = [0.020, 0.150]
    elif "SEF" in stimcat:
        par["ctrlwin"] = [-0.100, -0.010]
        par["actiwin"] = [0.010, 0.100]
    dfname_stimcat = dfname + "_" + stimcat

    epochs_stimcat = epochs[stimcat]

    # % % Find trial variance > index outliers> remove beyond plow and phigh percentile
    badtrls, plow, phigh = [], 2.0, 98.0
    bad_trials = my_var_cut_fn(epochs_stimcat, plow, phigh, to_plot=False)
    print(
        "\n%d trial to remove from total %d trials...\nNo. of remaining trials = %d\n"
        % (len(bad_trials), len(epochs_stimcat), len(epochs_stimcat) - len(bad_trials))
    )
    epochs_stimcat.drop(bad_trials, reason="variance based rejection", verbose=True)
    bad_trials = []

    # Compute covariance
    noise_cov = mne.compute_covariance(
        epochs_stimcat,
        tmin=par["ctrlwin"][0],
        tmax=par["ctrlwin"][1],
        method="empirical",
        verbose=True,
    )
    data_cov = mne.compute_covariance(
        epochs_stimcat,
        tmin=par["actiwin"][0],
        tmax=par["actiwin"][1],
        method="empirical",
        verbose=True,
    )

    evoked = epochs_stimcat.average()
    evoked = evoked.crop(par["actiwin"][0], par["actiwin"][1])

    # Pull rank from data preprocessing history
    cov_rank = (
        None
        if epochs_stimcat.info["proc_history"] == []
        else int(
            epochs_stimcat.info["proc_history"][0]["max_info"]["sss_info"]["nfree"]
        )
    )

    if False:
        # Compute SNR
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            evoked.info,
            fwd,
            noise_cov,
            rank=cov_rank,
            loose=1,
            depth=0.199,
            verbose=True,
        )
        snr, _ = mne.minimum_norm.estimate_snr(evoked, inverse_operator, verbose=True)
        peak_ch, peak_time = evoked.get_peak(ch_type="grad")
        tstep = 1000 / (evoked.info["sfreq"] * 1000)
        tp = int(peak_time // tstep - evoked.times[0] // tstep)
        SNR = snr[tp]

    # Compute filter and output
    filters = mne.beamformer.make_lcmv(
        epochs_stimcat.info,
        fwd,
        data_cov,
        reg=0.05,
        noise_cov=noise_cov,
        pick_ori="max-power",
        rank=cov_rank,
        weight_norm="nai",
        reduce_rank=True,
        verbose=True,
    )

    stc = mne.beamformer.apply_lcmv(evoked, filters)
    stc = np.abs(stc)
    src_peak, t_peak = stc.get_peak()
    timepoint = int(t_peak // stc.tstep - stc.times[0] // stc.tstep)

    # % % Detect peak over summed power over time:: Stable over time
    stc_pow_series = np.square(stc)
    stc_power = stc_pow_series.sum()

    src_peak, _ = stc_power.get_peak()
    est_loc = fwd["src"][0]["rr"][src_peak] * 1000

    # % % Calculate localization Error and Point spread volume
    loc_err = np.sqrt(np.sum(np.square(par["act_loc"][stimcat] - est_loc)))
    stc_power_data = stc_power.copy().data
    n_act_grid = len(stc_power_data[stc_power_data > (stc_power_data.max() * 0.50)])
    PSVol = n_act_grid * (5.0**3)

    print(
        "Act_Sourceloc for %s" % dfname_stimcat + "= %s" % str(par["act_loc"][stimcat])
    )
    print("Est_SourceLoc for %s" % dfname_stimcat + "= %s" % str(np.around(est_loc, 1)))
    print("Loc_error for %s" % dfname_stimcat + "= %.1f mm" % loc_err)
    print("Point Spread Volume (PSV) for %s" % dfname_stimcat + "= %.1f mm" % PSVol)

    # % % Plot the activation
    img = stc.as_volume(fwd["src"], dest="mri", mri_resolution=False, format="nifti1")
    plot_stat_map(index_img(img, timepoint), mrifile, threshold=stc.data.max() * 0.50)
    plt.suptitle(
        "%s\n" % dfname_stimcat
        + "PeakValue= %.3f, " % stc.data.max()
        + "Est_loc= [%.1f,%.1f,%.1f], " % tuple(est_loc)
        + "Loc_err= %.2f mm" % loc_err,
        fontsize=12,
        color="white",
    )

if do_rhino:
    out_nii_fname = op.join(
        subjects_dir, subject, "rhino", "power_{}mm.nii.gz".format(gridstep)
    )
    out_nii_fname, stdbrain_mask_fname = rhino.recon_timeseries2niftii(
        subjects_dir,
        subject,
        recon_timeseries=stc_pow_series.data,
        out_nii_fname=out_nii_fname,
        reference_brain="mri",
        times=epochs.times,
    )

    rhino.fsleyes_overlay(stdbrain_mask_fname, out_nii_fname)
