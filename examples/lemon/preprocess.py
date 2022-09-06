"""Example script for preprocessing LEMON.

In this script we use custom automated ICA functions.
"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import logging

import mne
import osl
import numpy as np
from scipy import io
from dask.distributed import Client

logger = logging.getLogger("osl")

RAW_DIR = "/ohba/pi/knobre/datasets/MBB-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"
OUT_DIR = "/ohba/pi/knobre/cgohil/lemon/preproc"

config = """
    preproc:
      - lemon_set_channel_montage: None
      - lemon_create_heog: None
      - set_channel_types: {VEOG: eog, HEOG: eog}
      - crop: {tmin: 10}
      - filter: {l_freq: 0.25, h_freq: 125, method: 'iir', iir_params: {order: 5, ftype: butter}}
      - notch_filter: {freqs: 50 100}
      - bad_channels: {picks: 'eeg'}
      - annotate_flat: None
      - resample: {sfreq: 250}
      - bad_segments: {segment_len: 2500, picks: 'eog'}
      - lemon_ica: {n_components: 30, picks: eeg}
      - bad_segments: {segment_len: 500, picks: 'eeg'}
      - bad_segments: {segment_len: 500, picks: 'eeg', mode: 'diff'}
      - drop_channels: {ch_names: ['VEOG', 'HEOG', 'ICA-VEOG', 'ICA-HEOG']}
      - interpolate_bads: None
      - set_eeg_reference: None
      - compute_current_source_density: None
      - filter: {l_freq: 0.5, h_freq: 30, method: 'iir', iir_params: {order: 5, ftype: butter}}
"""


def lemon_set_channel_montage(dataset, userargs):

    subj = "010060"
    ref_file = os.path.join(RAW_DIR, f"sub-{subj}", "RSEEG", f"sub-{subj}.mat")
    X = io.loadmat(ref_file)
    ch_pos = {}
    for ii in range(len(X["Channel"][0]) - 1):  # final channel is reference
        key = X["Channel"][0][ii][0][0].split("_")[2]
        if key[:2] == "FP":
            key = "Fp" + key[2]
        value = X["Channel"][0][ii][3][:, 0]
        value = np.array([value[1], value[0], value[2]])
        ch_pos[key] = value

    dig = mne.channels.make_dig_montage(ch_pos=ch_pos)
    dataset["raw"].set_montage(dig)

    return dataset


def lemon_create_heog(dataset, userargs):

    F7 = dataset["raw"].get_data(picks="F7")
    F8 = dataset["raw"].get_data(picks="F8")

    heog = F7 - F8

    info = mne.create_info(["HEOG"], dataset["raw"].info["sfreq"], ["eog"])
    eog_raw = mne.io.RawArray(heog, info)
    dataset["raw"].add_channels([eog_raw], force_update_info=True)

    return dataset


def lemon_ica(dataset, userargs, logfile=None):

    ica = mne.preprocessing.ICA(
        n_components=userargs["n_components"], max_iter=1000, random_state=42
    )

    fraw = dataset["raw"].copy().filter(l_freq=1.0, h_freq=None)

    ica.fit(fraw, picks=userargs["picks"])
    dataset["ica"] = ica

    logger.info("Starting EOG autoreject")

    # Find and exclude VEOG
    veog_indices, eog_scores = dataset["ica"].find_bads_eog(dataset["raw"], "VEOG")
    dataset["veog_scores"] = eog_scores
    dataset["ica"].exclude.extend(veog_indices)

    logger.info(
        "Marking {0} ICs as EOG {1}".format(len(dataset["ica"].exclude), veog_indices)
    )

    # Find and exclude HEOG
    heog_indices, eog_scores = dataset["ica"].find_bads_eog(dataset["raw"], "HEOG")
    dataset["heog_scores"] = eog_scores
    dataset["ica"].exclude.extend(heog_indices)

    logger.info("Marking {0} ICs as HEOG {1}".format(len(heog_indices), heog_indices))

    # Save components as channels in raw object
    src = dataset["ica"].get_sources(fraw).get_data()
    veog = src[veog_indices[0], :]
    heog = src[heog_indices[0], :]

    ica.labels_["top"] = [veog_indices[0], heog_indices[0]]

    info = mne.create_info(
        ["ICA-VEOG", "ICA-HEOG"], dataset["raw"].info["sfreq"], ["misc", "misc"]
    )
    eog_raw = mne.io.RawArray(np.c_[veog, heog].T, info)
    dataset["raw"].add_channels([eog_raw], force_update_info=True)

    # Apply ICA denoising or not
    if ("apply" not in userargs) or (userargs["apply"] is True):
        logger.info("Removing selected components from raw data")
        dataset["ica"].apply(dataset["raw"])
    else:
        logger.info("Components were not removed from raw data")

    return dataset


if __name__ == "__main__":
    osl.utils.logger.set_up(level="INFO")

    # Get input files
    raw_data_files = osl.utils.Study(RAW_DIR + "/{subject}/RSEEG/{subject}.vhdr")
    inputs = sorted(raw_data_files.get())

    # Setup parallel processing
    client = Client(n_workers=8, threads_per_worker=1)

    # Main preprocessing
    osl.preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=OUT_DIR,
        overwrite=True,
        extra_funcs=[lemon_set_channel_montage, lemon_create_heog, lemon_ica],
        dask_client=True,
    )
