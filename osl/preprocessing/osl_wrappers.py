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

def gesd(x, alpha=0.05, p_out=.1, outlier_side=0):
    """Detect outliers using Generalized ESD test

     Parameters
     ----------
     x : vector
        Data set containing outliers
     alpha : scalar
        Significance level to detect at (default = 0.05)
     p_out : int
        Maximum number of outliers to detect (default = 10% of data set)
     outlier_side : {-1,0,1}
        Specify sidedness of the test
           - outlier_side = -1 -> outliers are all smaller
           - outlier_side = 0  -> outliers could be small/negative or large/positive (default)
           - outlier_side = 1  -> outliers are all larger

    Returns
    -------
    idx : boolean vector
        Boolean array with TRUE wherever a sample is an outlier
    x2 : vector
        Input array with outliers removed

    References
    ----------
    B. Rosner (1983). Percentage Points for a Generalized ESD Many-Outlier Procedure. Technometrics 25(2), pp. 165-172.
    http://www.jstor.org/stable/1268549?seq=1

    """

    if outlier_side == 0:
        alpha = alpha/2

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    n_out = int(np.ceil(len(x)*p_out))

    if np.any(np.isnan(x)):
        # Need to find outliers only in finite x
        y = np.where(np.isnan(x))[0]
        idx1, x2 = gesd(x[np.isfinite(x)], alpha, n_out, outlier_side)

        # idx1 has the indexes of y which were marked as outliers
        # the value of y contains the corresponding indexes of x that are outliers
        idx = np.zeros_like(x).astype(bool)
        idx[y[idx1]] = True

    n = len(x)
    temp = x.copy()
    R = np.zeros((n_out,))
    rm_idx = np.zeros((n_out,), dtype=int)
    lam = np.zeros((n_out,))

    for j in range(0, int(n_out)):
        i = j+1
        if outlier_side == -1:
            rm_idx[j] = np.nanargmin(temp)
            sample = np.nanmin(temp)
            R[j] = np.nanmean(temp) - sample
        elif outlier_side == 0:
            rm_idx[j] = int(np.nanargmax(abs(temp-np.nanmean(temp))))
            R[j] = np.nanmax(abs(temp-np.nanmean(temp)))
        elif outlier_side == 1:
            rm_idx[j] = np.nanargmax(temp)
            sample = np.nanmax(temp)
            R[j] = sample - np.nanmean(temp)

        R[j] = R[j] / np.nanstd(temp)
        temp[int(rm_idx[j])] = np.nan

        p = 1-alpha/(n-i+1)
        t = stats.t.ppf(p, n-i-1)
        lam[j] = ((n-i) * t) / (np.sqrt((n-i-1+t**2)*(n-i+1)))

    # Create a boolean array of outliers
    idx = np.zeros((n,)).astype(bool)
    idx[rm_idx[np.where(R > lam)[0]]] = True

    x2 = x[~idx]

    return idx, x2

def _find_outliers_in_dims(X, axis=-1, metric_func=np.std, gesd_args=None):
    """Find outliers across specified dimensions of an array"""

    if gesd_args is None:
        gesd_args = {}

    if axis == -1:
        axis = np.arange(X.ndim)[axis]

    squashed_axes = tuple(np.setdiff1d(np.arange(X.ndim), axis))
    metric = metric_func(X, axis=squashed_axes)

    rm_ind, _ = gesd(metric, **gesd_args)

    return rm_ind


def _find_outliers_in_segments(X, axis=-1, segment_len=100,
                               metric_func=np.std, gesd_args=None):
    """Create dummy-segments in a dimension of an array and find outliers in it"""

    if gesd_args is None:
        gesd_args = {}

    if axis == -1:
        axis = np.arange(X.ndim)[axis]

    # Prepare to slice data array
    slc = []
    for ii in range(X.ndim):
        if ii == axis:
            slc.append(slice(0, segment_len))
        else:
            slc.append(slice(None))

    # Preallocate some variables
    starts = np.arange(0, X.shape[axis], segment_len)
    metric = np.zeros((len(starts), ))
    bad_inds = np.zeros(X.shape[axis])*np.nan

    # Main loop
    for ii in range(len(starts)):
        if ii == len(starts)-1:
            stop = None
        else:
            stop = starts[ii]+segment_len

        # Update slice on dim of interest
        slc[axis] = slice(starts[ii], stop)
        # Compute metric for current chunk
        metric[ii] = metric_func(X[tuple(slc)])
        # Store which chunk we've used
        bad_inds[slc[axis]] = ii

    # Get bad segments
    rm_ind, _ = gesd(metric, **gesd_args)
    # Convert to int indices
    rm_ind = np.where(rm_ind)[0]
    # Convert to bool in original space of defined axis
    bads = np.isin(bad_inds, rm_ind)
    return bads


def detect_artefacts(X, axis=None, reject_mode='dim', metric_func=np.std,
                     segment_len=100, gesd_args=None, ret_mode='bad_inds'):
    """Detect bad observations or segments in a dataset

    Parameters
    ----------
    X : ndarray
        Array to find artefacts in.
    axis : int
        Index of the axis to detect artefacts in
    reject_mode : {'dim' | 'segments'}
        Flag indicating whether to detect outliers across a dimension (dim;
        default) or whether to split a dim into segments and detect outliers in
        the them (segments)
    metric_func : function
        Function defining metric to detect outliers on. Defaults to np.std but
        can be any function taking an array and returning a single number.
    segement_len : int > 0
        Integer window length of dummy epochs for bad_segment detection
    gesd_args : dict
        Dictionary of arguments to pass to gesd
    ret_mode : {'good_inds','bad_inds','zero_bads','nan_bads'}
        Flag indicating whether to return the indices for good observations,
        indices for bad observations (default), the input data with outliers
        removed (zero_bads) or the input data with outliers replaced with nans
        (nan_bads)

    Returns
    -------
    ndarray
        If ret_mode is ``'bad_inds'`` or ``'good_inds'``, this returns a boolean vector
        of length ``X.shape[axis]`` indicating good or bad samples. If ``ret_mode`` is
        ``'zero_bads'`` or ``'nan_bads'`` this returns an array copy of the input data
        ``X`` with bad samples set to zero or ``np.nan`` respectively.

    """

    if reject_mode not in ['dim', 'segments']:
        raise ValueError("reject_mode: '{0}' not recognised".format(reject_mode))

    if ret_mode not in ['bad_inds', 'good_inds', 'zero_bads', 'nan_bads']:
        raise ValueError("ret_mode: '{0}' not recognised")

    if axis is None or axis > X.ndim:
        raise ValueError('bad axis')

    if reject_mode == 'dim':
        bad_inds = _find_outliers_in_dims(X, axis=axis, metric_func=metric_func, gesd_args=gesd_args)

    elif reject_mode == 'segments':
        bad_inds = _find_outliers_in_segments(X, axis=axis,
                                              segment_len=segment_len,
                                              metric_func=metric_func,
                                              gesd_args=gesd_args)

    if ret_mode == 'bad_inds':
        return bad_inds
    elif ret_mode == 'good_inds':
        return bad_inds == False  # noqa: E712
    elif ret_mode in ['zero_bads', 'nan_bads']:
        out = X.copy()

        slc = []
        for ii in range(X.ndim):
            if ii == axis:
                slc.append(bad_inds)
            else:
                slc.append(slice(None))
        slc = tuple(slc)

        if ret_mode == 'zero_bads':
            out[slc] = 0
            return out
        elif ret_mode == 'nan_bads':
            out[slc] = np.nan
            return out


def detect_maxfilt_zeros(raw):
    """This function tries to load the maxfilter log files in order 
        to annotate zeroed out data in the :py:class:`mne.io.Raw <mne.io.Raw>` object. It 
        assumes that the log file is in the same directory as the
        raw file and has the same name, but with the extension ``.log``.

    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE raw object.

    Returns
    -------
    bad_inds : np.array of bool (n_times,) or None
        Boolean array indicating which time points are zeroed out.
    """    
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
    """Set bad segments in an MNE :py:class:`Raw <mne.io.Raw>` object as defined by the Generalized ESD test in :py:func:`osl.preprocessing.osl_wrappers.gesd <osl.preprocessing.osl_wrappers.gesd>`.


    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE raw object.
    picks : str
        Channel types to pick. See Notes for recommendations.
    segment_len : int
        Window length to divide the data into (non-overlapping).
    significance_level : float
        Significance level for detecting outliers. Must be between 0-1.
    metric : str
        Metric to use. Could be ``'std'``, ``'var'`` or ``'kurtosis'``.
    ref_meg : str
        ref_meg argument to pass with :py:func:`mne.pick_types <mne.pick_types>`.
    mode : str
        Should be ``None`` ``'diff'`` or ``'maxfilter'``.
        When ``mode='diff'`` we calculate a difference time series before
        detecting bad segments. When ``mode='maxfilter'`` we only mark the
        segments with zeros from MaxFiltering as bad.
    detect_zeros : bool
        Should we detect segments of zeros based on the maxfilter files?

    Returns
    -------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE raw object with bad segments annotated.
        
    Notes
    -----
    Note that for Elekta/MEGIN data, we recommend using ``picks: 'mag'`` or ``picks: 'grad'`` separately (in no particular order).
    
    Note that with CTF data, mne.pick_types will return:
        ~274 axial grads (as magnetometers) if ``{picks: 'mag', ref_meg: False}``
        ~28 reference axial grads if ``{picks: 'grad'}``.
        Thus, it is recommended to use ``picks:'mag'`` in combination with ``ref_mag: False``, and ``picks:'grad'`` separately (in no particular order).
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
    elif picks == "emg":
        chinds = mne.pick_types(raw.info, emg=True, ref_meg=ref_meg, exclude='bads')
    elif picks == "misc":
        chinds = mne.pick_types(raw.info, misc=True, exclude='bads')
    else:
        raise NotImplementedError(f"picks={picks} not available.")

    if mode is None:
        if detect_zeros:
            bdinds_maxfilt = detect_maxfilt_zeros(raw)
        else:
            bdinds_maxfilt = None
        XX, XX_times = raw.get_data(picks=chinds, reject_by_annotation='omit', return_times=True)
    elif mode == "diff":
        bdinds_maxfilt = None
        XX, XX_times = raw.get_data(picks=chinds, reject_by_annotation='omit', return_times=True)
        XX = np.diff(XX, axis=1)
        XX_times = XX_times[1:] # remove the first time point
    elif mode == "maxfilter":
        bdinds_maxfilt = detect_maxfilt_zeros(raw)
        XX, XX_times = raw.get_data(picks=chinds, reject_by_annotation='omit', return_times=True)

    allowed_metrics = ["std", "var", "kurtosis"]
    if metric not in allowed_metrics:
        raise ValueError(f"metric {metric} unknown.")
    if metric == "std":
        metric_func = np.std
    elif metric == "var":
        metric_func = np.var
    else:
        def kurtosis(inputs):
            return stats.kurtosis(inputs, axis=None)
        metric_func = kurtosis
    
    if mode == "maxfilter":
        bad_indices = [bdinds_maxfilt]
    else:
        bdinds = detect_artefacts(
            XX,
            axis=1,
            reject_mode="segments",
            metric_func=metric_func,
            segment_len=segment_len,
            ret_mode="bad_inds",
            gesd_args=gesd_args,
        )
        bad_indices = [bdinds, bdinds_maxfilt]

    for count, bdinds in enumerate(bad_indices):
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
        descriptions = np.repeat("{0}bad_segment_{1}".format(descp1, picks), len(onsets))
        logger.info("Found {0} bad segments".format(len(onsets)))

        onsets_secs = raw.first_samp/raw.info["sfreq"] + XX_times[onsets.astype(int)]
        offsets_secs = raw.first_samp/raw.info["sfreq"] + XX_times[offsets.astype(int)]
        durations_secs = offsets_secs - onsets_secs

        raw.annotations.append(onsets_secs, durations_secs, descriptions)

        mod_dur = durations_secs.sum()
        full_dur = raw.n_times / raw.info["sfreq"]
        pc = (mod_dur / full_dur) * 100
        s = "Modality {0}{1} - {2:02f}/{3} seconds rejected     ({4:02f}%)"
        logger.info(s.format("picks", descp2, mod_dur, full_dur, pc))

    return raw


def detect_badchannels(raw, picks, ref_meg="auto", significance_level=0.05):
    """Set bad channels in an MNE :py:class:`Raw <mne.io.Raw>` object as defined by the Generalized ESD test in :py:func:`osl.preprocessing.osl_wrappers.gesd <osl.preprocessing.osl_wrappers.gesd>`.

    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE raw object.
    picks : str
        Channel types to pick. See Notes for recommendations.
    ref_meg : str
        ref_meg argument to pass with :py:func:`mne.pick_types <mne.pick_types>`.
    significance_level : float
        Significance level for detecting outliers. Must be between 0-1.

    Returns
    -------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object with bad channels marked.
        
    Notes
    -----
    Note that for Elekta/MEGIN data, we recommend using ``picks:'mag'`` or ``picks:'grad'`` separately (in no particular order).
    
    Note that with CTF data, mne.pick_types will return:
        ~274 axial grads (as magnetometers) if ``{picks: 'mag', ref_meg: False}``
        ~28 reference axial grads if ``{picks: 'grad'}``.
        Thus, it is recommended to use ``picks:'mag'`` in combination with ``ref_mag: False``,  and ``picks:'grad'`` separately (in no particular order).
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
    elif picks == "misc":
        chinds = mne.pick_types(raw.info, misc=True, exclude='bads')
    else:
        raise NotImplementedError(f"picks={picks} not available.")
    ch_names = np.array(raw.ch_names)[chinds]

    bdinds = detect_artefacts(
        raw.get_data(picks=chinds),
        axis=0,
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
    """Drop bad epochs in an MNE :py:class:`Epochs <mne.Epochs>` object as defined by the Generalized ESD test in :py:func:`osl.preprocessing.osl_wrappers.gesd <osl.preprocessing.osl_wrappers.gesd>`.
    
    Parameters
    ----------
    epochs : :py:class:`mne.Epochs <mne.Epochs>`
        MNE Epochs object.
    picks : str
        Channel types to pick.
    significance_level : float
        Significance level for detecting outliers. Must be between 0-1.
    max_percentage : float
        Maximum fraction of the epochs to drop. Should be between 0-1.
    outlier_side : int
        Specify sidedness of the test:
        
        * outlier_side = -1 -> outliers are all smaller
        
        * outlier_side = 0  -> outliers could be small/negative or large/positive (default)
        
        * outlier_side = 1  -> outliers are all larger
        
    metric : str
        Metric to use. Could be ``'std'``, ``'var'`` or ``'kurtosis'``.
    ref_meg : str
        ref_meg argument to pass with :py:func:`mne.pick_types <mne.pick_types>`.
    mode : str
        Should be ``'diff'`` or ``None``. When ``mode='diff'`` we calculate a difference time
        series before detecting bad segments.

    Returns
    -------
    epochs : :py:meth:`mne.Epochs <mne.Epochs>`
        MNE Epochs object with bad epoches marked.
        
    Notes
    -----
    Note that with CTF data, mne.pick_types will return:
        ~274 axial grads (as magnetometers) if ``{picks: 'mag', ref_meg: False}``
        ~28 reference axial grads if ``{picks: 'grad'}``.
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
    elif picks == "misc":
        chinds = mne.pick_types(epochs.info, misc=True, ref_meg=ref_meg, exclude='bads')
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
    bad_epochs, _ = gesd(X, **gesd_args)
    logger.info(
        f"Modality {picks} - {np.sum(bad_epochs)}/{X.shape[0]} epochs rejected"
    )

    # Drop bad epochs
    epochs.drop(bad_epochs)

    return epochs


# Wrapper functions


def run_osl_bad_segments(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`detect_badsegments <osl.preprocessing.osl_wrappers.detect_badsegments>`.
    
    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`detect_badsegments <osl.preprocessing.osl_wrappers.detect_badsegments>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0} : {1}".format(target, "detect_badsegments"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset["raw"] = detect_badsegments(dataset["raw"], **userargs)
    return dataset


def run_osl_bad_channels(dataset, userargs):
    """OSL-Batch wrapper for :py:func:`detect_badchannels <osl.preprocessing.osl_wrappers.detect_badchannels>`.
    
    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`detect_badchannels <osl.preprocessing.osl_wrappers.detect_badchannels>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
        
    Notes 
    -----
    Note that using 'picks' with CTF data, mne.pick_types will return:
        ~274 axial grads (as magnetometers) if ``{picks: 'mag', ref_meg: False}``
        ~28 reference axial grads if ``{picks: 'grad'}``.
    """

    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0} : {1}".format(target, "detect_badchannels"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset["raw"] = detect_badchannels(dataset["raw"], **userargs)
    return dataset


def run_osl_drop_bad_epochs(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`drop_bad_epochs <osl.preprocessing.osl_wrappers.drop_bad_epochs>`.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`drop_bad_epochs <osl.preprocessing.osl_wrappers.drop_bad_epochs>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0} : {1}".format(target, "detect_bad_epochs"))
    logger.info("userargs: {0}".format(str(userargs)))
    if dataset["epochs"] is None:
        logger.info("no epoch object found! skipping..")
    dataset["epochs"] = drop_bad_epochs(dataset["epochs"], **userargs)
    return dataset


def run_osl_ica_manualreject(dataset, userargs):
    """OSL-Batch wrapper for :py:func:`osl.preprocessing.plot_ica <osl.preprocessing.plot_ica.plot_ica>`, and optionally :py:meth:`ICA.apply <mne.preprocessing.ICA.apply>`.

    This function opens an interactive window to allow the user to manually reject ICA components. Note that this function will modify the input MNE object in place. 
    The interactive plotting function might not work on all systems depending on the backend in use. It will most likely work when using an IDE (e.g. Spyder, 
    Pycharm, VS Cose) but not in a Jupyter Notebook.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the keys ``raw`` and ``ica``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:func:`plot_ica <osl.preprocessing.plot_ica>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
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
