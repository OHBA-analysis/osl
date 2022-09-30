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

logger = logging.getLogger(__name__)


# --------------------------------------------------------------
# OSL preprocessing functions
#

def detect_maxfilt_zeros(raw):
    """This function tries to load the maxfilter log files in order to annotate zeroed out data"""
    log_fname = raw.filenames[0].replace('.fif', '.log')
    if exists(log_fname):
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
            logger.error(s)
    else:
        s = "No maxfilter logfile detected - detecting zeroed out data not possible"
        logger.info(s)
        return


def detect_badsegments(raw, segment_len=1000, picks="grad", mode=None):
    """Set bad segments in MNE object."""
    if mode is None:
        bdinds_maxfilt = detect_maxfilt_zeros(raw)
        XX = raw.get_data(picks=picks)
    elif mode == "diff":
        bdinds_maxfilt = None
        XX = np.diff(raw.get_data(picks=picks), axis=1)

    bdinds_std = sails.utils.detect_artefacts(
        XX, 1, reject_mode="segments", segment_len=segment_len, ret_mode="bad_inds"
    )
    for count, bdinds in enumerate([bdinds_std, bdinds_maxfilt]):
        if count == 1:
            descp = 'maxfilter_'
        else:
            descp = ''
        onsets = np.where(np.diff(bdinds.astype(float)) == 1)[0]
        if bdinds[0]:
            onsets = np.r_[0, onsets]
        offsets = np.where(np.diff(bdinds.astype(float)) == -1)[0]

        if bdinds[-1]:
            offsets = np.r_[offsets, len(bdinds) - 1]
        assert len(onsets) == len(offsets)
        durations = offsets - onsets
        descriptions = np.repeat("{0}bad_segment_{1}".format(descp, picks), len(onsets))
        logger.info("Found {0} bad segments".format(len(onsets)))

        onsets = (onsets + raw.first_samp) / raw.info["sfreq"]
        durations = durations / raw.info["sfreq"]

        raw.annotations.append(onsets, durations, descriptions)

        mod_dur = durations.sum()
        full_dur = raw.n_times / raw.info["sfreq"]
        pc = (mod_dur / full_dur) * 100
        s = "Modality {0}{1} - {2:02f}/{3} seconds rejected     ({4:02f}%)"
        logger.info(s.format("picks", descp.replace('m',' (m').replace('_',')'), mod_dur, full_dur, pc))

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
