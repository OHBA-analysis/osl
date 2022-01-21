#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

import mne
import numpy as np

from ..utils import osl_print

# --------------------------------------------------------------
# MNE Raw/Epochs Object Methods
#
# We have run_mne_anonymous which tries to run a method directly on a target
# object (typocally an mne Raw or Epochs object.
#
# In addition, there are a set of wrapper functions for MNE methods which need
# a bit more option processing than the default - for example, converting input
# strings into arrays of frequencies
#
# Wrapper functions have priority and will be run rather than the direct method
# call if a wrapper is present. If no wrapper is present then we fall back to
# the direct method call.

def run_mne_anonymous(dataset, userargs, method, logfile=None):
    osl_print('\nMNE ANON - {0}'.format(method), logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    target = userargs.pop('target', 'raw')
    if hasattr(dataset[target], method) and callable(getattr(dataset[target], method)):
        getattr(dataset[target], method)(**userargs)
    else:
        raise ValueError("Method '{0}' not found on target '{1}'".format(method, target))
    return dataset


# --------------------------------------------------------------
# Exceptions for which we can't just naively pass args through

# General wrapper functions - work on Raw and Epochs

def run_mne_notch_filter(dataset, userargs, logfile=None):
    osl_print('\nNOTCH-FILTERING', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    target = userargs.pop('target', 'raw')
    freqs = np.array(userargs.pop('freqs').split(' ')).astype(float)
    dataset[target].notch_filter(freqs, **userargs)
    return dataset


def run_mne_pick(dataset, userargs, logfile=None):
    osl_print('\nPICKING', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    target = userargs.pop('target', 'raw')
    dataset[target].pick(**userargs)
    return dataset


def run_mne_pick_channels(dataset, userargs, logfile=None):
    osl_print('\nPICKING CHANNELS ', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    target = userargs.pop('target', 'raw')
    dataset[target].pick_channels(**userargs)
    return dataset


def run_mne_pick_types(dataset, userargs, logfile=None):
    osl_print('\nPICKING CHANNEL TYPES', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    target = userargs.pop('target', 'raw')
    dataset[target].pick_types(**userargs)
    return dataset


def run_mne_resample(dataset, userargs, logfile=None):
    """The MNE guys don't seem to like resampling so much..."""
    osl_print('\nRESAMPLING', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    target = userargs.pop('target', 'raw')
    if ('events' in dataset) and (dataset['events'] is not None):
        dataset[target], dataset['events'] = dataset[target].resample(events=dataset['events'], **userargs)
    else:
        dataset[target].resample(**userargs)
    return dataset


# Epochs Functions

def run_mne_drop_bad(dataset, userargs, logfile=None):
    osl_print('\nMNE DROP BAD', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    target = userargs.pop('target', 'epochs')  # should only be epochs

    # Need to sanitise values in 'reject' dictionary - these are strings after
    # being read in from yaml.
    if 'reject' in userargs:
        for key, value in userargs['reject'].items():
            userargs['reject'][key] = float(value)

    dataset[target] = dataset[target].drop_bad(**userargs)

    return dataset


# TFR 

def run_mne_apply_baseline(dataset, userargs, logfile=None):
    osl_print('\nAPPLY BASELINE', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    target = userargs.pop('target', 'raw')
    freqs = np.array(userargs.pop('baseline').split(' ')).astype(float)
    dataset[target].apply_baseline(freqs, **userargs)
    return dataset


# --------------------------------------------------------------
# MNE Events and Epochs Object Methods
#

# These wrappers use MNE functions which use one or more different data types
# and typically create a new data object in the dataset dictionary.

def run_mne_find_events(dataset, userargs, logfile=None):
    osl_print('\nFINDING EVENTS', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    dataset['events'] = mne.find_events(dataset['raw'], **userargs)
    return dataset


def run_mne_epochs(dataset, userargs, logfile=None):
    osl_print('\nEPOCHING', logfile=logfile)
    osl_print(userargs, logfile=logfile)
    tmin = userargs.pop('tmin', -0.2)
    tmax = userargs.pop('tmax', 0.5)
    dataset['epochs'] = mne.Epochs(dataset['raw'],
                                   dataset['events'],
                                   dataset['event_id'],
                                   tmin, tmax, **userargs)
    return dataset


# Time-frequency transforms


def run_mne_tfr_multitaper(dataset, userargs, logfile=None):
    osl_print('\nTFR MULTITAPER', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    target = userargs.pop('target', 'epochs')
    from mne.time_frequency import tfr_multitaper

    freqs = np.array(userargs.pop('freqs').split(' ')).astype('float')
    freqs = np.linspace(freqs[0], freqs[1], int(freqs[2]))
    out = tfr_multitaper(dataset[target],
                         freqs,
                         **userargs)
    if 'return_itc' in userargs and userargs['return_itc']:
        dataset['power'], dataset['itc'] = out
    else:
        dataset['power'] = out

    return dataset


def run_mne_tfr_morlet(dataset, userargs, logfile=None):
    osl_print('\nTFR MORLET', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    target = userargs.pop('target', 'epochs')
    from mne.time_frequency import tfr_morlet

    freqs = np.array(userargs.pop('freqs').split(' ')).astype('float')
    reqs = np.linspace(freqs[0], freqs[1], int(freqs[2]))
    dataset['power'], dataset['itc'] = tfr_morlet(dataset[target],
                                                  freqs,
                                                  **userargs)

    return dataset


def run_mne_tfr_stockwell(dataset, userargs, logfile=None):
    osl_print('\nTFR STOCKWELL', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    target = userargs.pop('target', 'epochs')
    from mne.time_frequency import tfr_stockwell

    out = tfr_stockwell(dataset[target], **userargs)
    if 'return_itc' in userargs and userargs['return_itc']:
        dataset['power'], dataset['itc'] = out
    else:
        dataset['power'] = out

    return dataset

# --------------------------------------------------------------
# OHBA/MNE ICA Tools
#
# Currently assuming ICA on Raw data - no Epochs support yet


def run_mne_ica_raw(dataset, userargs, logfile=None):
    osl_print('\nICA', logfile=logfile)
    # NOTE: **userargs doesn't work because 'picks' is in there
    ica = mne.preprocessing.ICA(n_components=userargs['n_components'])
    ica.fit(dataset['raw'], picks=userargs['picks'])
    dataset['ica'] = ica
    return dataset


def run_mne_ica_autoreject(dataset, userargs, logfile=None):
    osl_print('\nICA AUTOREJECT', logfile=logfile)
    if np.logical_or('ecgmethod' not in userargs, userargs['ecgmethod'] == 'ctps'):
        ecgmethod = 'ctps'
    elif userargs['ecgmethod'] == 'correlation':
        ecgmethod = 'correlation'
    if ecgmethod == 'ctps':
        ecgthreshold = 'auto'
    elif ecgmethod == 'correlation':
        ecgthreshold = 3
    eog_indices, eog_scores = dataset['ica'].find_bads_eog(dataset['raw'],
                                                           threshold=0.35,
                                                           measure='correlation')

    dataset['ica'].exclude.extend(eog_indices)
    osl_print('Marking {0} as EOG ICs'.format(len(dataset['ica'].exclude)), logfile=logfile)
    ecg_indices, ecg_scores = dataset['ica'].find_bads_ecg(dataset['raw'],
                                                           threshold=ecgthreshold,
                                                           method=ecgmethod)
    dataset['ica'].exclude.extend(ecg_indices)
    osl_print('Marking {0} as ECG ICs'.format(len(dataset['ica'].exclude)), logfile=logfile)
    if ('apply' not in userargs) or (userargs['apply'] is True):
        osl_print('\nREMOVING SELECTED COMPONENTS FROM RAW DATA', logfile=logfile)
        dataset['ica'].apply(dataset['raw'])
    else:
        osl_print('\nCOMPONENTS WERE NOT REMOVED FROM RAW DATA', logfile=logfile)
    return dataset


def run_osl_ica_manualreject(dataset, userargs):
    print('\nICA MANUAL REJECT')
    from .osl_plot_ica import plot_ica
    plot_ica(dataset['ica'], dataset['raw'], block=True)
    print('Removing {0} IC'.format(len(dataset['ica'].exclude)))
    if np.logical_or('apply' not in userargs, userargs['apply'] is True):
        print('\nREMOVING SELECTED COMPONENTS FROM RAW DATA')
        dataset['ica'].apply(dataset['raw'])
    else:
        print('\nCOMPONENTS WERE NOT REMOVED FROM RAW DATA')
    return dataset


def run_mne_apply_ica(dataset, userargs, logfile=None):
    osl_print('\nAPPLYING ICA', logfile=logfile)
    dataset['raw'] = dataset['ica'].apply(dataset['raw'])
    return dataset
