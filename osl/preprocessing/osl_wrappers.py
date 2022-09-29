"""Wrappers for MNE functions to perform preprocessing.

"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>

import logging
import mne
import numpy as np
import sails

logger = logging.getLogger(__name__)


# --------------------------------------------------------------
# OSL preprocessing functions
#


def detect_badsegments(raw, segment_len=1000, picks="grad", mode=None):
    """Set bad segments in MNE object."""
    if mode is None:
        XX = raw.get_data(picks=picks)
    elif mode == "diff":
        XX = np.diff(raw.get_data(picks=picks), axis=1)

    bdinds = sails.utils.detect_artefacts(
        XX, 1, reject_mode="segments", segment_len=segment_len, ret_mode="bad_inds"
    )

    onsets = np.where(np.diff(bdinds.astype(float)) == 1)[0]
    if bdinds[0]:
        onsets = np.r_[0, onsets]
    offsets = np.where(np.diff(bdinds.astype(float)) == -1)[0]

    if bdinds[-1]:
        offsets = np.r_[offsets, len(bdinds) - 1]
    assert len(onsets) == len(offsets)
    durations = offsets - onsets
    descriptions = np.repeat("bad_segment_{0}".format(picks), len(onsets))
    logger.info("Found {0} bad segments".format(len(onsets)))

    onsets = (onsets + raw.first_samp) / raw.info["sfreq"]
    durations = durations / raw.info["sfreq"]

    raw.annotations.append(onsets, durations, descriptions)

    mod_dur = durations.sum()
    full_dur = raw.n_times / raw.info["sfreq"]
    pc = (mod_dur / full_dur) * 100
    s = "Modality {0} - {1:02f}/{2} seconds rejected     ({3:02f}%)"
    logger.info(s.format("picks", mod_dur, full_dur, pc))

    return raw


def detect_badchannels(raw, picks="grad"):
    """Set bad channels in MNE object."""
    bdinds = sails.utils.detect_artefacts(
        raw.get_data(picks=picks), 0, reject_mode="dim", ret_mode="bad_inds"
    )

    if (picks == "mag") or (picks == "grad"):
        chinds = mne.pick_types(raw.info, meg=picks)
    elif picks == "meg":
        chinds = mne.pick_types(raw.info, meg=True)
    elif picks == "eeg":
        chinds = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = np.array(raw.ch_names)[chinds]

    s = "Modality {0} - {1}/{2} channels rejected     ({3:02f}%)"
    pc = (bdinds.sum() / len(bdinds)) * 100
    logger.info(s.format(picks, bdinds.sum(), len(bdinds), pc))

    if np.any(bdinds):
        raw.info["bads"] = list(ch_names[np.where(bdinds)[0]])

    return raw


# Wrapper functions


def run_osl_bad_segments(dataset, userargs, logfile=None):
    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0} : {1}".format(target, "detect_badsegments"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset["raw"] = detect_badsegments(dataset["raw"], **userargs)
    return dataset


def run_osl_bad_channels(dataset, userargs, logfile=None):
    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0} : {1}".format(target, "detect_badchannels"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset["raw"] = detect_badchannels(dataset["raw"], **userargs)
    return dataset


def run_osl_ica_manualreject(dataset, userargs):
    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0}".format("ICA Manual Reject"))
    logger.info("userargs: {0}".format(str(userargs)))

    from .plot_ica import plot_ica

    plot_ica(dataset["ica"], dataset["raw"], block=True)
    logger.info("Removing {0} IC".format(len(dataset["ica"].exclude)))
    if np.logical_or("apply" not in userargs, userargs["apply"] is True):
        logger.info("Removing selected components from raw data")
        dataset["ica"].apply(dataset["raw"])
    else:
        logger.info("Components were not removed from raw data")
    return dataset