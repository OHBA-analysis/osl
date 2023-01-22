"""Wrappers for MNE functions to perform preprocessing.

"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>

import mne
import numpy as np

# Housekeeping for logging
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# MNE Raw/Epochs Object Methods
#
# We have run_mne_anonymous which tries to run a method directly on a target
# object (typically an mne Raw or Epochs object).
#
# In addition, there are a set of wrapper functions for MNE methods which need
# a bit more option processing than the default - for example, converting input
# strings into arrays of frequencies
#
# Wrapper functions have priority and will be run rather than the direct method
# call if a wrapper is present. If no wrapper is present then we fall back to
# the direct method call.


def run_mne_anonymous(dataset, userargs, method):
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, method))
    logger.info("userargs: {0}".format(str(userargs)))
    if hasattr(dataset[target], method) and callable(getattr(dataset[target], method)):
        getattr(dataset[target], method)(**userargs)
    else:
        raise ValueError(
            "Method '{0}' not found on target '{1}'".format(method, target)
        )
    return dataset


# --------------------------------------------------------------
# Exceptions for which we can't just naively pass args through

# General wrapper functions - work on Raw and Epochs


def run_mne_notch_filter(dataset, userargs):
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "notch_filter"))
    logger.info("userargs: {0}".format(str(userargs)))
    freqs = userargs.pop("freqs")
    freqs = [
        float(freqs)
        if np.logical_or(type(freqs) == int, type(freqs) == float)
        else np.array(freqs.split(" ")).astype(float)
    ]
    dataset[target].notch_filter(freqs, **userargs)
    return dataset


def run_mne_pick(dataset, userargs):
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "pick"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset[target].pick(**userargs)
    return dataset


def run_mne_pick_channels(dataset, userargs):
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "pick_channels"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset[target].pick_channels(**userargs)
    return dataset


def run_mne_pick_types(dataset, userargs):
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "pick_types"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset[target].pick_types(**userargs)
    return dataset


def run_mne_resample(dataset, userargs):
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "resample"))
    logger.info("userargs: {0}".format(str(userargs)))
    if ("events" in dataset) and (dataset["events"] is not None):
        dataset[target], dataset["events"] = dataset[target].resample(
            events=dataset["events"], **userargs
        )
    else:
        dataset[target].resample(**userargs)
    return dataset


def run_mne_set_channel_types(dataset, userargs):
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "set_channel_types"))
    logger.info("userargs: {0}".format(str(userargs)))
    # Separate function as we don't explode userargs
    dataset[target].set_channel_types(userargs)
    return dataset


# Epochs Functions


def run_mne_drop_bad(dataset, userargs):
    target = userargs.pop("target", "epochs")  # should only be epochs
    logger.info("MNE Stage - {0}.{1}".format(target, "drop_bad"))
    logger.info("userargs: {0}".format(str(userargs)))

    # Need to sanitise values in 'reject' dictionary - these are strings after
    # being read in from yaml.
    if "reject" in userargs:
        for key, value in userargs["reject"].items():
            userargs["reject"][key] = float(value)

    dataset[target] = dataset[target].drop_bad(**userargs)

    return dataset


# TFR


def run_mne_apply_baseline(dataset, userargs):
    target = userargs.pop("target", "epochs")  # should only be epochs
    logger.info("MNE Stage - {0}.{1}".format(target, "apply_baseline"))
    logger.info("userargs: {0}".format(str(userargs)))

    freqs = np.array(userargs.pop("baseline").split(" ")).astype(float)
    dataset[target].apply_baseline(freqs, **userargs)
    return dataset


# --------------------------------------------------------------
# MNE Events and Epochs Object Methods
#

# These wrappers use MNE functions which use one or more different data types
# and typically create a new data object in the dataset dictionary.


def run_mne_find_events(dataset, userargs):
    logger.info("MNE Stage - {0}.{1}".format("mne", "find_events"))
    logger.info("userargs: {0}".format(str(userargs)))

    dataset["events"] = mne.find_events(dataset["raw"], **userargs)
    return dataset


def run_mne_epochs(dataset, userargs):
    logger.info("MNE Stage - {0}.{1}".format("mne", "epochs"))
    logger.info("userargs: {0}".format(str(userargs)))
    tmin = userargs.pop("tmin", -0.2)
    tmax = userargs.pop("tmax", 0.5)
    dataset["epochs"] = mne.Epochs(
        dataset["raw"], dataset["events"], dataset["event_id"], tmin, tmax, **userargs
    )
    return dataset


# --------------------------------------------------------------
# mne.preprocessing functions


def run_mne_annotate_flat(dataset, userargs):
    logger.info("MNE Stage - {0}.{1}".format("mne.preprocessing", "annotate_flat"))
    logger.info("userargs: {0}".format(str(userargs)))
    target = userargs.pop("target", "raw")

    # Import func - otherwise line is too long even for me
    from mne.preprocessing import annotate_flat

    bad_annotations, flat_channels = annotate_flat(dataset[target], **userargs)

    dataset[target].info["bads"].extend(flat_channels)

    # Can't combine annotations with different orig_times, the following line
    # fails if uncommented >>
    # dataset[target].set_annotations(dataset[target].annotations + bad_annotations)

    # ...so I'm extracting and reforming the muscle annotations here. Feels like
    # the line above would be better if it worked - must be missing something?
    onsets = []
    durations = []
    descriptions = []
    for ann in bad_annotations:
        onsets.append(ann["onset"])
        durations.append(ann["duration"])
        descriptions.append(ann["description"])

    dataset[target].annotations.append(onsets, durations, descriptions)

    return dataset


def run_mne_annotate_muscle_zscore(dataset, userargs):
    logger.info(
        "MNE Stage - {0}.{1}".format("mne.preprocessing", "annotate_muscle_zscore")
    )
    logger.info("userargs: {0}".format(str(userargs)))
    target = userargs.pop("target", "raw")

    # Import func - otherwise line is too long even for me
    from mne.preprocessing import annotate_muscle_zscore

    bad_annotations, _ = annotate_muscle_zscore(dataset[target], **userargs)

    # Can't combine annotations with different orig_times, the following line
    # fails if uncommented >>
    # dataset[target].set_annotations(dataset[target].annotations + bad_annotations)

    # ...so I'm extracting and reforming the muscle annotations here. Feels like
    # the line above would be better if it worked.
    onsets = []
    durations = []
    descriptions = []
    for ann in bad_annotations:
        onsets.append(ann["onset"])
        durations.append(ann["duration"])
        descriptions.append(ann["description"])

    dataset[target].annotations.append(onsets, durations, descriptions)

    return dataset


def run_mne_find_bad_channels_maxwell(dataset, userargs):
    logger.info(
        "MNE Stage - {0}.{1}".format("mne.preprocessing", "find_bad_channels_maxwell")
    )
    logger.info("userargs: {0}".format(str(userargs)))
    target = userargs.pop("target", "raw")

    # Import func - otherwise line is too long even for me
    from mne.preprocessing import find_bad_channels_maxwell

    # Run maxfilter
    noisy, flats, scores = find_bad_channels_maxwell(dataset[target], **userargs)

    dataset[target].info["bads"].extend(noisy + flat)

    return dataset


def run_mne_maxwell_filter(dataset, userargs):
    logger.info("MNE Stage - {0}.{1}".format("mne.preprocessing", "maxwell_filter"))
    logger.info("userargs: {0}".format(str(userargs)))
    target = userargs.pop("target", "raw")

    # Import func - otherwise line is too long even for me
    from mne.preprocessing import maxwell_filter

    # Run maxfilter
    dataset[target] = maxwell_filter(dataset[target], **userargs)

    return dataset


def run_mne_compute_current_source_density(dataset, userargs):
    logger.info(
        "MNE Stage - {0}.{1}".format(
            "mne.preprocessing", "compute_current_source_density"
        )
    )
    logger.info("userargs: {0}".format(str(userargs)))
    target = userargs.pop("target", "raw")

    # Import func - otherwise line is too long even for me
    from mne.preprocessing import compute_current_source_density

    # Run Laplacian
    dataset[target] = compute_current_source_density(dataset[target], **userargs)

    return dataset


# Time-frequency transforms


def run_mne_tfr_multitaper(dataset, userargs):
    target = userargs.pop("target", "epochs")
    logger.info("MNE Stage - {0} on {1}".format("tfr_multitaper", target))
    logger.info("userargs: {0}".format(str(userargs)))
    from mne.time_frequency import tfr_multitaper

    freqs = np.array(userargs.pop("freqs").split(" ")).astype("float")
    freqs = np.linspace(freqs[0], freqs[1], int(freqs[2]))
    out = tfr_multitaper(dataset[target], freqs, **userargs)
    if "return_itc" in userargs and userargs["return_itc"]:
        dataset["power"], dataset["itc"] = out
    else:
        dataset["power"] = out

    return dataset


def run_mne_tfr_morlet(dataset, userargs):
    target = userargs.pop("target", "epochs")
    logger.info("MNE Stage - {0} on {1}".format("tfr_morlet", target))
    logger.info("userargs: {0}".format(str(userargs)))
    from mne.time_frequency import tfr_morlet

    freqs = np.array(userargs.pop("freqs").split(" ")).astype("float")
    reqs = np.linspace(freqs[0], freqs[1], int(freqs[2]))
    dataset["power"], dataset["itc"] = tfr_morlet(dataset[target], freqs, **userargs)

    return dataset


def run_mne_tfr_stockwell(dataset, userargs):
    target = userargs.pop("target", "epochs")
    logger.info("MNE Stage - {0} on {1}".format("tfr_stockwell", target))
    logger.info("userargs: {0}".format(str(userargs)))
    from mne.time_frequency import tfr_stockwell

    out = tfr_stockwell(dataset[target], **userargs)
    if "return_itc" in userargs and userargs["return_itc"]:
        dataset["power"], dataset["itc"] = out
    else:
        dataset["power"] = out

    return dataset


# --------------------------------------------------------------
# OHBA/MNE ICA Tools
#
# Currently assuming ICA on Raw data - no Epochs support yet


def run_mne_ica_raw(dataset, userargs):
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}".format("mne.preprocessing.ICA"))
    logger.info("userargs: {0}".format(str(userargs)))
    
    # MNE recommends applying a high pass filter at 1 Hz before calculating
    # the ICA (make this adjustable by user):
    # https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#filtering-to-remove-slow-drifts
    l_freq = userargs.pop("l_freq", 1)
    h_freq = userargs.pop("h_freq", None)
    filt_raw = dataset["raw"].copy().filter(l_freq=l_freq, h_freq=h_freq)

    # NOTE: **userargs doesn't work because 'picks' is in there
    ica = mne.preprocessing.ICA(n_components=userargs["n_components"])
    ica.fit(filt_raw, picks=userargs["picks"])
    dataset["ica"] = ica
    return dataset


def run_mne_ica_autoreject(dataset, userargs):
    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0}".format("ICA Autoreject"))
    logger.info("userargs: {0}".format(str(userargs)))

    # User specified arguments and their defaults
    eogmeasure = userargs.pop("eogmeasure", "correlation")
    eogthreshold = userargs.pop("eogthreshold", 0.35)
    ecgmethod = userargs.pop("ecgmethod", "ctps")
    ecgthreshold = userargs.pop("ecgthreshold", "auto")
    remove_components = userargs.pop("apply", True)

    # Reject components based on the EOG channel
    eog_indices, eog_scores = dataset["ica"].find_bads_eog(
        dataset["raw"], threshold=eogthreshold, measure=eogmeasure,
    )
    dataset["ica"].exclude.extend(eog_indices)
    logger.info("Marking {0} as EOG ICs".format(len(eog_indices)))

    # Reject components based on the ECG channel
    ecg_indices, ecg_scores = dataset["ica"].find_bads_ecg(
        dataset["raw"], threshold=ecgthreshold, method=ecgmethod
    )
    dataset["ica"].exclude.extend(ecg_indices)
    logger.info("Marking {0} as ECG ICs".format(len(ecg_indices)))

    # Remove the components from the data if requested
    if remove_components:
        logger.info("Removing selected components from raw data")
        dataset["ica"].apply(dataset["raw"])
    else:
        logger.info("Components were not removed from raw data")

    return dataset


def run_mne_apply_ica(dataset, userargs):
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}".format("ica.apply"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset["raw"] = dataset["ica"].apply(dataset["raw"])
    return dataset
