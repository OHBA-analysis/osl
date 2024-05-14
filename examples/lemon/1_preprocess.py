"""Example script for preprocessing resting-state EEG data in the LEMON dataset.

This script preprocesses the data in parallel.
"""

# Authors : Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#         : Andrew Quinn <a.quinn@bham.ac.uk>

import logging
from glob import glob
from pathlib import Path

import mne
import numpy as np
from scipy import io
from dask.distributed import Client

from osl import preprocessing, utils

logger = logging.getLogger("osl")

# Directories
RAW_DIR = "/well/woolrich/projects/lemon/raw"
LOCALIZER_DIR = RAW_DIR + "/EEG_MPILMBB_LEMON/EEG_Localizer_BIDS_ID"
PREPROC_DIR = "/well/woolrich/projects/lemon/osl_example/preproc"

# Files
RAW_FILE = RAW_DIR + "/{0}/RSEEG/{0}.vhdr"
LOCALIZER_FILE = LOCALIZER_DIR + "/{0}/{0}.mat"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    def lemon_set_channel_montage(dataset, userargs):
        subject = Path(dataset["raw"]._filenames[0]).stem
        loc_file = LOCALIZER_FILE.format(subject)
        X = io.loadmat(loc_file, simplify_cells=True)
        ch_pos = {}
        for i in range(len(X["Channel"]) - 1):  # final channel is reference
            key = X["Channel"][i]["Name"].split("_")[2]
            if key[:2] == "FP":
                key = "Fp" + key[2]
            value = X["Channel"][i]["Loc"]
            ch_pos[key] = value
        hp = X["HeadPoints"]["Loc"]
        nas = np.mean([hp[:, 0], hp[:, 3]], axis=0)
        lpa = np.mean([hp[:, 1], hp[:, 4]], axis=0)
        rpa = np.mean([hp[:, 2], hp[:, 5]], axis=0)
        dig = mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=nas, lpa=lpa, rpa=rpa)
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

    # Settings
    config = """
        preproc:
          - lemon_set_channel_montage: {}
          - lemon_create_heog: {}
          - set_channel_types: {VEOG: eog, HEOG: eog}
          - crop: {tmin: 15}
          - filter: {l_freq: 0.25, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
          - notch_filter: {freqs: 50 100}
          - resample: {sfreq: 250}
          - bad_channels: {picks: eeg}
          - bad_segments: {segment_len: 2500, picks: eog}
          - bad_segments: {segment_len: 500, picks: eeg, significance_level: 0.1}
          - bad_segments: {segment_len: 500, picks: eeg, mode: diff, significance_level: 0.1}
          - lemon_ica: {n_components: 30, picks: eeg}
          - interpolate_bads: None
          - drop_channels: {ch_names: ['VEOG', 'HEOG', 'ICA-VEOG', 'ICA-HEOG']}
          - set_eeg_reference: {projection: true}
    """

    # Get names of subjects we have localizer data for
    subjects = []
    for loc_file in sorted(glob(LOCALIZER_DIR + "/sub-*")):
        subjects.append(Path(loc_file).name)
    subjects = subjects[:2]

    # Generate a list of input files
    inputs = []
    for subject in subjects:
        raw_file = Path(RAW_FILE.format(subject))
        if raw_file.exists():
            inputs.append(raw_file)

    # Setup parallel processing
    client = Client(n_workers=2, threads_per_worker=1)

    # Preprocessing
    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=PREPROC_DIR,
        overwrite=True,
        extra_funcs=[lemon_set_channel_montage, lemon_create_heog, lemon_ica],
        dask_client=True,
    )
