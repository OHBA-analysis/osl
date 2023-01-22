"""Calculate first-level GLM.

"""

import os
import osl
import mne
import h5py
import sails
import numpy as np
from glob import glob
from scipy import stats
from dask.distributed import Client


def get_device_fids(raw):
    # Put fiducials in device space
    head_fids = mne.viz._3d._fiducial_coords(raw.info["dig"])
    head_fids = np.vstack(([0, 0, 0], head_fids))
    fid_space = raw.info["dig"][0]["coord_frame"]
    assert fid_space == 4  # Ensure we have FIFFV_COORD_HEAD coords

    # Get device to head transform and inverse
    dev2head = raw.info["dev_head_t"]
    head2dev = mne.transforms.invert_transform(dev2head)
    assert head2dev["from"] == 4
    assert head2dev["to"] == 1

    # Apply transformation to get fids in device space
    device_fids = mne.transforms.apply_trans(head2dev, head_fids)

    return device_fids


def make_bads_regressor(raw, mode="raw"):
    bads = np.zeros((raw.n_times,))
    for an in raw.annotations:
        if an["description"].startswith("bad") and an["description"].endswith(mode):
            start = raw.time_as_index(an["onset"])[0] - raw.first_samp
            duration = int(an["duration"] * raw.info["sfreq"])
            bads[start : start + duration] = 1
    if mode == "raw":
        bads[: int(raw.info["sfreq"] * 2)] = 1
        bads[-int(raw.info["sfreq"] * 2) :] = 1
    else:
        bads[: int(raw.info["sfreq"])] = 1
        bads[-int(raw.info["sfreq"]) :] = 1
    return bads


def run_first_level(dataset, userargs):
    run_id = osl.utils.find_run_id(dataset["raw"].filenames[0])
    subj_id = run_id.split("_")[0]

    # Bad segments
    bads = make_bads_regressor(dataset["raw"], mode="mag")

    # EOGs - vertical only
    eogs = dataset["raw"].copy().pick_types(meg=False, eog=True)
    eogs = eogs.filter(l_freq=1, h_freq=20, picks="eog").get_data()

    # ECG - lots of sanity checking
    ecg_events, ch_ecg, av_pulse = mne.preprocessing.find_ecg_events(dataset["raw"])
    ecg_events[:, 0] = ecg_events[:, 0] - dataset["raw"].first_samp
    ecg = np.zeros_like(bads)
    median_beat = np.median(np.diff(ecg_events[:, 0]))
    last_beat = median_beat
    for ii in range(ecg_events.shape[0] - 1):
        beat = ecg_events[ii + 1, 0] - ecg_events[ii, 0]
        if np.abs(beat - last_beat) > 50:
            beat = last_beat
        ecg[ecg_events[ii, 0] : ecg_events[ii + 1, 0]] = beat
        beat = last_beat
    ecg[ecg == 0] = median_beat
    ecg = ecg / dataset["raw"].info["sfreq"] * 60

    # Store covariates
    confs = {"VEOG": np.abs(eogs[0, :]), "BadSegs": bads}
    covs = {"Linear": np.linspace(-1, 1, ecg.shape[0]), "ECG": ecg}
    conds = None
    conts = None

    #%% -------------------------------------------
    # GLM - With z-transform

    # Get data
    XX = stats.zscore(dataset["raw"].get_data(picks="meg"))
    fs = dataset["raw"].info["sfreq"]

    # Fit model
    freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(
        XX,
        axis=1,
        fit_constant=True,
        conditions=conds,
        covariates=covs,
        confounds=confs,
        contrasts=conts,
        nperseg=int(fs * 2),
        noverlap=int(fs),
        fmin=0.1,
        fmax=100,
        fs=fs,
        mode="magnitude",
        fit_method="glmtools",
    )
    model, design, data = extras

    # Save out
    outdir = userargs.get("outdir")
    hdfname = os.path.join(
        outdir, "{subj_id}-glm-data-ztrans.hdf5".format(subj_id=run_id)
    )
    if os.path.exists(hdfname):
        print("Overwriting previous results")
        os.remove(hdfname)
    with h5py.File(hdfname, "w") as F:
        model.to_hdf5(F.create_group("model"))
        design.to_hdf5(F.create_group("design"))
        # data.to_hdf5(F.create_group('data'))
        F.create_dataset("freq_vect", data=freq_vect)
        F.create_dataset("device_fids", data=get_device_fids(dataset["raw"]))
        F.create_dataset("scan_duration", data=dataset["raw"].times[-1])
        F.create_dataset("av_bpm", data=av_pulse)

    #%% -------------------------------------------
    # GLM - No z-transform

    # Get data
    XX = dataset["raw"].get_data(picks="meg")
    fs = dataset["raw"].info["sfreq"]

    # Fit model
    freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(
        XX,
        axis=1,
        fit_constant=True,
        conditions=conds,
        covariates=covs,
        confounds=confs,
        contrasts=conts,
        nperseg=int(fs * 2),
        noverlap=int(fs),
        fmin=0.1,
        fmax=100,
        fs=fs,
        mode="magnitude",
        fit_method="glmtools",
    )
    model, design, data = extras

    # Save out
    outdir = userargs.get("outdir")
    hdfname = os.path.join(
        outdir, "{subj_id}-glm-data-notrans.hdf5".format(subj_id=run_id)
    )
    if os.path.exists(hdfname):
        print("Overwriting previous results")
        os.remove(hdfname)
    with h5py.File(hdfname, "w") as F:
        model.to_hdf5(F.create_group("model"))
        design.to_hdf5(F.create_group("design"))
        # data.to_hdf5(F.create_group('data'))
        F.create_dataset("freq_vect", data=freq_vect)
        F.create_dataset("device_fids", data=get_device_fids(dataset["raw"]))
        F.create_dataset("scan_duration", data=dataset["raw"].times[-1])
        F.create_dataset("av_bpm", data=av_pulse)

    # Always need to return the dataset - even though its unchanged in this func
    return dataset


config = """
    preproc:
    - run_first_level: {outdir: /well/woolrich/projects/camcan/winter23/glm}
"""

if __name__ == "__main__":
    inputs = sorted(
        glob(
            "/well/woolrich/projects/camcan/winter23/preproc"
            + "/mf2pt2_*_ses-rest_task-rest_meg"
            + "/mf2pt2_*_ses-rest_task-rest_meg_preproc_raw.fif"
        )
    )

    client = Client(n_workers=16, threads_per_worker=1)

    osl.preprocessing.run_proc_batch(
        config,
        inputs,
        outdir="./tmp",  # this directory can be deleted after running
        extra_funcs=[run_first_level],
        overwrite=True,
        dask_client=True,
    )
