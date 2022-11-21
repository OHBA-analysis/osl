"""Wrappers for MNE functions to perform preprocessing.

"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>

import logging
import mne
import numpy as np
import sails
from os.path import exists
from scipy import stats

logger = logging.getLogger(__name__)


# --------------------------------------------------------------
# OSL preprocessing functions
#

def detect_maxfilt_zeros(raw):
    """This function tries to load the maxfilter log files in order to annotate zeroed out data"""
    if raw.filenames[0] is not None:
        log_fname = raw.filenames[0].replace('.fif', '.log')
    if 'log_fname' in locals() and exists(log_fname):
        try:
            starttime = raw.first_time
            endtime = raw._last_time
            with open(log_fname) as f:
                lines = f.readlines()

            # for determining the start, end and  point
            phrase_ndataseg = ['(', ' data buffers)']
            gotduration = False

            # for detecting zeroed out data
            zeroed=[]
            phrase_zero = ['Time ', ': cont HPI is off, data block is skipped!']
            for line in lines:
                if gotduration == False and phrase_ndataseg[1] in line:
                    gotduration = True
                    n_dataseg = float(line.split(phrase_ndataseg[0])[1].split(phrase_ndataseg[1])[0]) # number of segments
                if phrase_zero[1] in line:
                    zeroed.append(float(line.split(phrase_zero[0])[1].split(phrase_zero[1])[0])) # in seconds

            duration = raw.n_times/n_dataseg # duration of each data segment in samples
            starts = (np.array(zeroed) - starttime) * raw.info['sfreq'] # in samples
            bad_inds = np.zeros(raw.n_times)
            for ii in range(len(starts)):
                stop = starts[ii] + duration  # in samples
                bad_inds[int(starts[ii]):int(stop)] = 1
            return bad_inds.astype(bool)
        except:
            s = "detecting zeroed out data from maxfilter log file failed"
            logger.warning(s)
            return None
    else:
        s = "No maxfilter logfile detected - detecting zeroed out data not possible"
        logger.info(s)
        return None


def detect_badsegments(
    raw,
    picks,
    segment_len=1000,
    significance_level=0.05,
    metric='std',
    ref_meg='auto',
    mode=None,
    detect_zeros=True,
):
    """Set bad segments in MNE object.

    Note that with CTF data, mne.pick_types will return:
    ~274 axial grads (as magnetometers) if {picks: 'mag', ref_meg: False}
    ~28 reference axial grads if {picks: 'grad'}
    """

    gesd_args = {'alpha': significance_level}

    if (picks == "mag") or (picks == "grad"):
        chinds = mne.pick_types(raw.info, meg=picks, ref_meg=ref_meg, exclude='bads')
    elif picks == "meg":
        chinds = mne.pick_types(raw.info, meg=True, ref_meg=ref_meg, exclude='bads')
    elif picks == "eeg":
        chinds = mne.pick_types(raw.info, eeg=True, ref_meg=ref_meg, exclude='bads')
    elif picks == "eog":
        chinds = mne.pick_types(raw.info, eog=True, ref_meg=ref_meg, exclude='bads')
    elif picks == "ecg":
        chinds = mne.pick_types(raw.info, ecg=True, ref_meg=ref_meg, exclude='bads')
    else:
        raise NotImplementedError(f"picks={picks} not available.")

    if mode is None:
        if detect_zeros:
            bdinds_maxfilt = detect_maxfilt_zeros(raw)
        else:
            bdinds_maxfilt = None
        XX = raw.get_data(picks=chinds)
    elif mode == "diff":
        bdinds_maxfilt = None
        XX = np.diff(raw.get_data(picks=chinds), axis=1)

    allowed_metrics = ["std", "var", "kurtosis"]
    if metric not in allowed_metrics:
        raise ValueError(f"metric {metric} unknown.")
    if metric == "std":
        metric_func = np.std
    elif metric == "var":
        metric_func = np.var
    else:
        metric_func = stats.kurtosis

    bdinds = sails.utils.detect_artefacts(
        XX,
        axis=1,
        reject_mode="segments",
        metric_func=metric_func,
        segment_len=segment_len,
        ret_mode="bad_inds",
        gesd_args=gesd_args,
    )
    for count, bdinds in enumerate([bdinds, bdinds_maxfilt]):
        if bdinds is None:
            continue
        if count==1:
            descp1 = count * 'maxfilter_' # when count==0, should be ''
            descp2 = ' (maxfilter)'
        else:
            descp1 = ''
            descp2 = ''
        onsets = np.where(np.diff(bdinds.astype(float)) == 1)[0]
        if bdinds[0]:
            onsets = np.r_[0, onsets]
        offsets = np.where(np.diff(bdinds.astype(float)) == -1)[0]

        if bdinds[-1]:
            offsets = np.r_[offsets, len(bdinds) - 1]
        assert len(onsets) == len(offsets)
        durations = offsets - onsets
        descriptions = np.repeat("{0}bad_segment_{1}".format(descp1, picks), len(onsets))
        logger.info("Found {0} bad segments".format(len(onsets)))

        onsets = (onsets + raw.first_samp) / raw.info["sfreq"]
        durations = durations / raw.info["sfreq"]

        raw.annotations.append(onsets, durations, descriptions)

        mod_dur = durations.sum()
        full_dur = raw.n_times / raw.info["sfreq"]
        pc = (mod_dur / full_dur) * 100
        s = "Modality {0}{1} - {2:02f}/{3} seconds rejected     ({4:02f}%)"
        logger.info(s.format("picks", descp2, mod_dur, full_dur, pc))

    return raw


def detect_badchannels(raw, picks, ref_meg="auto", significance_level=0.05):
    """Set bad channels in MNE object.

    Note that with CTF data, mne.pick_types will return:
    ~274 axial grads (as magnetometers) if {picks: 'mag', ref_meg: False}
    ~28 reference axial grads if {picks: 'grad'}
    """

    gesd_args = {'alpha': significance_level}

    if (picks == "mag") or (picks == "grad"):
        chinds = mne.pick_types(raw.info, meg=picks, ref_meg=ref_meg, exclude='bads')
    elif picks == "meg":
        chinds = mne.pick_types(raw.info, meg=True, ref_meg=ref_meg, exclude='bads')
    elif picks == "eeg":
        chinds = mne.pick_types(raw.info, eeg=True, ref_meg=ref_meg, exclude='bads')
    elif picks == "eog":
        chinds = mne.pick_types(raw.info, eog=True, ref_meg=ref_meg, exclude='bads')
    elif picks == "ecg":
        chinds = mne.pick_types(raw.info, ecg=True, ref_meg=ref_meg, exclude='bads')
    else:
        raise NotImplementedError(f"picks={picks} not available.")
    ch_names = np.array(raw.ch_names)[chinds]

    bdinds = sails.utils.detect_artefacts(
        raw.get_data(picks=chinds),
        0,
        reject_mode="dim",
        ret_mode="bad_inds",
        gesd_args=gesd_args,
    )

    s = "Modality {0} - {1}/{2} channels rejected     ({3:02f}%)"
    pc = (bdinds.sum() / len(bdinds)) * 100
    logger.info(s.format(picks, bdinds.sum(), len(bdinds), pc))

    # concatenate newly found bads to existing bads
    if np.any(bdinds):
        raw.info["bads"].extend(list(ch_names[np.where(bdinds)[0]]))

    return raw


def drop_bad_epochs(
    epochs,
    picks,
    significance_level=0.05,
    max_percentage=0.1,
    outlier_side=0,
    metric='std',
    ref_meg='auto',
    mode=None,
):
    """Drop bad epochs in MNE object.

    Note that with CTF data, mne.pick_types will return:
    ~274 axial grads (as magnetometers) if {picks: 'mag', ref_meg: False}
    ~28 reference axial grads if {picks: 'grad'}
    """

    gesd_args = {
        'alpha': significance_level,
        'p_out': max_percentage,
        'outlier_side': outlier_side,
    }

    if (picks == "mag") or (picks == "grad"):
        chinds = mne.pick_types(epochs.info, meg=picks, ref_meg=ref_meg, exclude='bads')
    elif picks == "meg":
        chinds = mne.pick_types(epochs.info, meg=True, ref_meg=ref_meg, exclude='bads')
    elif picks == "eeg":
        chinds = mne.pick_types(epochs.info, eeg=True, ref_meg=ref_meg, exclude='bads')
    elif picks == "eog":
        chinds = mne.pick_types(epochs.info, eog=True, ref_meg=ref_meg, exclude='bads')
    elif picks == "ecg":
        chinds = mne.pick_types(epochs.info, ecg=True, ref_meg=ref_meg, exclude='bads')
    else:
        raise NotImplementedError(f"picks={picks} not available.")

    if mode is None:
        X = epochs.get_data(picks=chinds)
    elif mode == "diff":
        X = np.diff(epochs.get_data(picks=chinds), axis=-1)

    # Get the function used to calculate the evaluation metric
    allowed_metrics = ["std", "var", "kurtosis"]
    if metric not in allowed_metrics:
        raise ValueError(f"metric {metric} unknown.")
    if metric == "std":
        metric_func = np.std
    elif metric == "var":
        metric_func = np.var
    else:
        metric_func = stats.kurtosis

    # Calculate the metric used to evaluate whether an epoch is bad
    X = metric_func(X, axis=-1)

    # Average over channels so we have a metric for each trial
    X = np.mean(X, axis=1)

    # Use gesd to find outliers
    bad_epochs, _ = sails.utils.gesd(X, **gesd_args)
    logger.info(
        f"Modality {picks} - {np.sum(bad_epochs)}/{X.shape[0]} epochs rejected"
    )

    # Drop bad epochs
    epochs.drop(bad_epochs)

    return epochs


# Wrapper functions


def run_osl_bad_segments(dataset, userargs, logfile=None):
    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0} : {1}".format(target, "detect_badsegments"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset["raw"] = detect_badsegments(dataset["raw"], **userargs)
    return dataset


def run_osl_bad_channels(dataset, userargs, logfile=None):
    """
    Note that with CTF data, mne.pick_types will return:
    ~274 axial grads (as magnetometers) if {picks: 'mag', ref_meg: False}
    ~28 reference axial grads if {picks: 'grad'}
    """

    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0} : {1}".format(target, "detect_badchannels"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset["raw"] = detect_badchannels(dataset["raw"], **userargs)
    return dataset


def run_osl_drop_bad_epochs(dataset, userargs, logfile=None):
    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0} : {1}".format(target, "detect_bad_epochs"))
    logger.info("userargs: {0}".format(str(userargs)))
    if dataset["epochs"] is None:
        logger.info("no epoch object found! skipping..")
    dataset["epochs"] = drop_bad_epochs(dataset["epochs"], **userargs)
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
