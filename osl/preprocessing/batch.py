#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""Tools for batch preprocessing ephys data.

MNE Advice.

* L-poass filter before downsampling!
 - Makes a huge speed difference

* Use notch_filter rather than the manual bandstop option in standard filter

* Don't downsample?
https://mne.tools/dev/auto_tutorials/preprocessing/30_filtering_resampling.html#resampling
... or at least downsample events with data (this is default here)

XML sanity check on config file. pre-parser to avoid mad sequences of methods.
"""

import argparse
import csv
import multiprocessing as mp
import matplotlib.pyplot as plt
import os
import sys
import pathlib
import traceback
from copy import deepcopy
from functools import partial, wraps
from time import localtime, strftime
from datetime import datetime
import re

import mne
import numpy as np
import sails
import yaml
from joblib import Parallel, delayed

from ..utils import find_run_id, validate_outdir, process_file_inputs
from ..utils import logger as osl_logger
from . import _mne_wrappers

# Housekeeping for logging
import logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------
# Decorators


def print_custom_func_info(func):
    """Prints info for user-specified functions."""
    @wraps(func)
    def wrapper(dataset, userargs):
        logger.info("CUSTOM Stage - {}".format(func.__name__))
        logger.info("userargs: {}".format(str(userargs)))
        return func(dataset, userargs)
    return wrapper

# --------------------------------------------------------------
# Data importers


def import_data(infile, preload=True, logfile=None):
    if not isinstance(infile, str):
        raise ValueError(
            "infile must be a str. Got type(infile)={0}.".format(type(infile))
        )

    logger.info('IMPORTING: {0}'.format(infile))

    if os.path.split(infile)[1] == 'c,rfDC':
        logger.info('Detected BTI file format, using: mne.io.read_raw_bti')
        # We have a BTI scan
        if os.path.isfile(os.path.join(os.path.split(infile)[0], 'hs_file')):
            head_shape_fname = 'hs_file'
        else:
            head_shape_fname = None
        raw = mne.io.read_raw_bti(infile, head_shape_fname=head_shape_fname, preload=preload)
    elif os.path.splitext(infile)[1] == '.fif':
        logger.info('Detected fif file format, using: mne.io.read_raw_fif')
        # We have a FIF file
        raw = mne.io.read_raw_fif(infile, preload=preload)
    elif os.path.splitext(infile)[1] == '.meg4':
        logger.info('Detected CTF file format, using: mne.io.read_raw_ctf')
        # We have the meg file from a ds directory
        raw = mne.io.read_raw_ctf(os.path.dirname(infile), preload=preload)
        logger.info('Detected CTF file format, using: mne.io.read_raw_ctf')
    elif os.path.splitext(infile)[1] == '.ds':
        raw = mne.io.read_raw_ctf(infile, preload=preload)
    elif os.path.splitext(infile)[1] == '.vhdr':
        logger.info('Detected brainvision file format, using: mne.io.read_raw_brainvision')
        raw = mne.io.read_raw_brainvision(infile, preload=preload)
    else:
        msg = 'Unable to determine file type of input {0}'.format(infile)
        logger.error(msg)
        raise ValueError(msg)

    return raw


# --------------------------------------------------------------
# OHBA Preprocessing functions
#
# TODO - probably want to move the working functions their own OSL module
# similar to _mne_wrappers (with ICA functions?).


def detect_badsegments(raw, segment_len=1000, picks='grad', mode=None):
    """Set bad segments in MNE object."""
    if mode is None:
        XX = raw.get_data(picks=picks)
    elif mode == 'diff':
        XX = np.diff(raw.get_data(picks=picks), axis=1)

    bdinds = sails.utils.detect_artefacts(XX, 1,
                                          reject_mode='segments',
                                          segment_len=segment_len,
                                          ret_mode='bad_inds')

    onsets = np.where(np.diff(bdinds.astype(float)) == 1)[0]
    if bdinds[0]:
        onsets = np.r_[0, onsets]
    offsets = np.where(np.diff(bdinds.astype(float)) == -1)[0]

    if bdinds[-1]:
        offsets = np.r_[offsets, len(bdinds)-1]
    assert(len(onsets) == len(offsets))
    durations = offsets - onsets
    descriptions = np.repeat('bad_segment_{0}'.format(picks), len(onsets))
    logger.info('Found {0} bad segments'.format(len(onsets)))

    onsets = (onsets + raw.first_samp) / raw.info['sfreq']
    durations = durations / raw.info['sfreq']

    raw.annotations.append(onsets, durations, descriptions)

    mod_dur = durations.sum()
    full_dur = raw.n_times/raw.info['sfreq']
    pc = (mod_dur / full_dur) * 100
    s = 'Modality {0} - {1:02f}/{2} seconds rejected     ({3:02f}%)'
    logger.info(s.format('picks', mod_dur, full_dur, pc))

    return raw


def detect_badchannels(raw, picks='grad'):
    """Set bad channels in MNE object."""
    bdinds = sails.utils.detect_artefacts(raw.get_data(picks=picks), 0,
                                          reject_mode='dim',
                                          ret_mode='bad_inds')

    if (picks == 'mag') or (picks == 'grad'):
        chinds = mne.pick_types(raw.info, meg=picks)
    elif (picks == 'meg'):
        chinds = mne.pick_types(raw.info, meg=True)
    elif (picks == 'eeg'):
        chinds = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = np.array(raw.ch_names)[chinds]

    s = 'Modality {0} - {1}/{2} channels rejected     ({3:02f}%)'
    pc = (bdinds.sum() / len(bdinds)) * 100
    logger.info(s.format(picks, bdinds.sum(), len(bdinds), pc))

    if np.any(bdinds):
        raw.info['bads'] = list(ch_names[np.where(bdinds)[0]])

    return raw

# Wrapper functions

def run_osl_bad_segments(dataset, userargs, logfile=None):
    target = userargs.pop('target', 'raw')
    logger.info('OSL Stage - {0} : {1}'.format(target, 'detect_badsegments'))
    logger.info('userargs: {0}'.format(str(userargs)))
    dataset['raw'] = detect_badsegments(dataset['raw'], **userargs)
    return dataset


def run_osl_bad_channels(dataset, userargs, logfile=None):
    target = userargs.pop('target', 'raw')
    logger.info('OSL Stage - {0} : {1}'.format(target, 'detect_badchannels'))
    logger.info('userargs: {0}'.format(str(userargs)))
    dataset['raw'] = detect_badchannels(dataset['raw'], **userargs)
    return dataset


# --------------------------------------------------------------
# Bach processing utilities

def find_func(method, target='raw', extra_funcs=None):
    # Function priority
    # 1) User custom function
    # 2) MNE/OSL wrapper
    # 3) MNE method on Raw or Epochs (specified by target)

    func = None

    # 1) user custom function
    if extra_funcs is not None:
        func_ind = [idx if (f.__name__ == method) else -1 for idx, f in enumerate(extra_funcs)]
        if np.max(func_ind) > -1:
            func = extra_funcs[np.argmax(func_ind)]

    # 2) MNE/OSL Wrapper
    # Find OSL function in local module
    if func is None:
        func = globals().get('run_osl_{0}'.format(method))

    # Find MNE function in local module
    if func is None and hasattr(_mne_wrappers, 'run_mne_{0}'.format(method)):
        func = getattr(_mne_wrappers, 'run_mne_{0}'.format(method))

    # 3) MNE direct method
    if func is None:
        if target == 'raw':
            if hasattr(mne.io.Raw, method) and callable(getattr(mne.io.Raw, method)):
                func = partial(_mne_wrappers.run_mne_anonymous, method=method)
        elif target == 'epochs':
            if hasattr(mne.Epochs, method) and callable(getattr(mne.Epochs, method)):
                func = partial(_mne_wrappers.run_mne_anonymous, method=method)
        elif target in ('power', 'itc'):
            if hasattr(mne.time_frequency.EpochsTFR, method) and callable(getattr(mne.time_frequency.EpochsTFR, method)):
                func = partial(_mne_wrappers.run_mne_anonymous, method=method)

    if func is None:
        logger.critical('Func not found! {0}'.format(method))

    return func


def load_config(config):
    if isinstance(config, str):
        try:
            # See if we have a filepath
            with open(config, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            # We have a string
            config = yaml.load(config, Loader=yaml.FullLoader)
    elif isinstance(config, dict):
        # Its already a dict
        pass

    # Initialise missing values in config
    if 'meta' not in config:
        config['meta'] = {'event_codes': None}
    elif 'event_codes' not in config['meta']:
        config['meta']['event_codes'] = None

    # Validation
    if 'preproc' not in config:
        raise KeyError('Please specify preprocessing steps in config.')

    for stage in config['preproc']:
        # Check each stage is a dictionary with a single key
        if not isinstance(stage, dict):
            raise ValueError("Preprocessing stage '{0}' is a {1} not a dict".format(stage, type(stage)))

        if len(stage) != 1:
            raise ValueError("Preprocessing stage '{0}' should only have a single key".format(stage))

        for key, val in stage.items():
            # internally we want options to be an empty dict (for now at least)
            if val in ['null', 'None', None]:
                stage[key] = {}

    for step in config['preproc']:
        if config['meta']['event_codes'] is None and 'find_events' in step.values():
            raise KeyError(
                'event_codes must be passed in config if we are finding events.'
            )

    return config


def get_config_from_fif(data):

    config_list = re.findall('%% config start %%(.*?)%% config end %%', data.info['description'], flags=re.DOTALL)
    config=[]
    for config_text in config_list:
        config.append(load_config(config_text))

    return config


def append_preprocinfo(dataset, config):
    if dataset['raw'].info['description']==None:
        dataset['raw'].info['description']=''
    preprocinfo = f"\n\nOSL BATCH PROCESSING APPLIED ON {datetime.today().strftime('%d/%m/%Y %H:%M:%S')} \n%% config start %% \n{config} \n%% config end %%"
    dataset['raw'].info['description'] = dataset['raw'].info['description'] + preprocinfo

    if dataset['epochs'] is not None:
        if dataset['epochs'].info['description']==None:
            dataset['epochs'].info['description']=''
        dataset['epochs'].info['description'] = dataset['epochs'].info['description'] + preprocinfo

    return dataset


def write_dataset(dataset, outbase, run_id, overwrite=False):
    # Save output
    outname = outbase.format(run_id=run_id.replace('_raw', ''), ftype='preproc_raw', fext='fif')
    if pathlib.Path(outname).exists() and not overwrite:
        raise ValueError('{} already exists. Please delete or do use overwrite=True.'.format(outname))
    dataset['raw'].save(outname, overwrite=overwrite)

    if dataset['events'] is not None:
        outname = outbase.format(run_id=run_id, ftype='events', fext='npy')
        np.save(outname, dataset['events'])

    if dataset['epochs'] is not None:
        outname = outbase.format(run_id=run_id, ftype='epo', fext='fif')
        dataset['epochs'].save(outname, overwrite=overwrite)

    if dataset['ica'] is not None:
        outname = outbase.format(run_id=run_id, ftype='ica', fext='fif')
        dataset['ica'].save(outname, overwrite=overwrite)


def plot_preproc_flowchart(config, outname=None, show=True, stagecol='wheat', startcol='red', fig=None, ax=None, title=None):
    """Make a summary flowchart of a preprocessing chain."""
    config = load_config(config)
    
    if np.logical_or(ax==None, fig==None):
        fig = plt.figure(figsize=(8, 12))
        plt.subplots_adjust(top=0.95, bottom=0.05)
        ax = plt.subplot(111, frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    if title==None:
        ax.set_title('OSL Preprocessing Recipe', fontsize=24)
    else:
        ax.set_title(title, fontsize=24)

    stage_height = 1/(1+len(config['preproc']))

    box = dict(boxstyle='round', facecolor=stagecol, alpha=1, pad=0.3)
    startbox = dict(boxstyle='round', facecolor=startcol, alpha=1)
    font = {'family': 'serif',
            'color':  'k',
            'weight': 'normal',
            'size': 16,
            }

    stages = [{'input': ''}, *config['preproc'], {'output': ''}]
    stage_str = "$\\bf{{{0}}}$ {1}"

    ax.arrow( 0.5, 1, 0.0, -1, fc="k", ec="k",
    head_width=0.045, head_length=0.035, length_includes_head=True)

    for idx, stage in enumerate(stages):

        method, userargs = next(iter(stage.items()))

        method = method.replace('_', '\_')
        if method in ['input', 'output']:
            b = startbox
        else:
            b = box
            method = method + ':'

        ax.text(0.5, 1-stage_height*idx, stage_str.format(method, str(userargs)[1:-1]),
                ha='center', va='center', bbox=b, fontdict=font, wrap=True)

    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.25, 0.75)

    if outname is not None:
        fig.savefig(outname, dpi=300, transparent=True)

    if show is True:
        fig.show()

    return fig, ax


# --------------------------------------------------------------
# Bach processing

def run_proc_chain(infile, config, outname=None, outdir=None, ret_dataset=True,
                   overwrite=False, extra_funcs=None, verbose='INFO', mneverbose='WARNING'):

    if outname is None:
        run_id = find_run_id(infile)
    else:
        run_id = os.path.splitext(outname)[0]

    if not isinstance(config, dict):
        config = load_config(config)

    if outdir is not None:
        name_base = '{run_id}_{ftype}.{fext}'
        outbase = os.path.join(outdir, name_base)
        logfile = outbase.format(run_id=run_id.replace('_raw', ''), ftype='preproc_raw', fext='log')
        mne.utils._logging.set_log_file(logfile)
    else:
        logfile = None
    osl_logger.set_up(prefix=run_id, log_file=logfile, level=verbose, startup=False)
    mne.set_log_level(mneverbose)
    logger = logging.getLogger(__name__)
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    logger.info('{0} : Starting OSL Processing'.format(now))
    logger.info('input : {0}'.format(infile))

    if isinstance(infile, str):
        raw = import_data(infile)
    elif isinstance(infile, mne.io.fiff.raw.Raw):
        raw = infile
        infile = raw.filenames[0]  # assuming only one file here

    dataset = {'raw': raw,
               'ica': None,
               'epochs': None,
               'events': None,
               'event_id': config['meta']['event_codes']}

    for stage in deepcopy(config['preproc']):
        method, userargs = next(iter(stage.items()))  # next(iter( feels a bit clumsy..
        target = userargs.get('target', 'raw')  # Raw is default
        func = find_func(method, target=target, extra_funcs=extra_funcs)
        try:
            if func in extra_funcs:
                # This is a custom function, let's print its info
                func = print_custom_func_info(func)
            dataset = func(dataset, userargs)
        except Exception as e:
            logger.critical('PROCESSING FAILED!!!!')
            ex_type, ex_value, ex_traceback = sys.exc_info()
            logger.error("{0} : {1}".format(method, func))
            logger.error(ex_type)
            logger.error(ex_value)
            logger.error(traceback.print_tb(ex_traceback))
            if outdir is not None:
                with open(logfile.replace('.log', '.error.log'), 'w') as f:
                    f.write('Processing filed during stage : "{0}"'.format(method))
                    f.write(str(ex_type))
                    f.write('\n')
                    f.write(str(ex_value))
                    f.write('\n')
                    traceback.print_tb(ex_traceback, file=f)
            return 0

    dataset = append_preprocinfo(dataset, config)

    if outdir is not None:
        write_dataset(dataset, outbase, run_id, overwrite=overwrite)

    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    logger.info('{0} : Processing Complete'.format(now))

    if ret_dataset:
        return dataset
    else:
        return 1


def run_proc_batch(config, files, outdir=None, overwrite=False, extra_funcs=None,
                   nprocesses=1, verbose='INFO', mneverbose='WARNING'):
    """
    files can be a list of Raw objects or a list of filenames or a path to a
    textfile list of filenames

    Currently writes outputs to disk - does not return a list of processed
    data! this could potentially be huge amounts of memory.

    """

    # -------------------------------------------------------------
    if outdir is None:
        # Use the current working directory
        outdir = os.getcwd()
    outdir = validate_outdir(outdir)

    mne.set_log_level(mneverbose)
    logfile = os.path.join(outdir, 'osl_batch.log')
    osl_logger.set_up(log_file=logfile, level=verbose, startup=False)

    logger.info('Starting OSL Batch Processing')

    infiles, outnames, good_files = process_file_inputs(files)
    logger.info('Processing {0} files'.format(sum(good_files)))

    logger.info('Outputs saving to: {0}\n\n'.format(outdir))
    config = load_config(config)

    # -------------------------------------------------------------
    # Create partial function with fixed options
    pool_func = partial(run_proc_chain,
                        outdir=outdir,
                        ret_dataset=False,
                        overwrite=overwrite,
                        extra_funcs=extra_funcs)

    # For each file...
    args = []
    for idx, infif in enumerate(infiles):
        if outnames is None:
            outname = None
        else:
            outname = outnames[idx]

        args.append((infif, config, outname))

    # Actually run the processes
    with Parallel(n_jobs=nprocesses, verbose=50) as parallel:
        proc_flags = parallel(delayed(pool_func)(*aa) for aa in args)

    logger.info('Processed {0}/{1} files successfully'.format(np.sum(proc_flags), len(proc_flags)))

    # Return failed flags
    return proc_flags


# ----------------------------------------------------------
# Main CLI user function


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Batch preprocess some fif files.')
    parser.add_argument('config', type=str,
                        help='yaml defining preproc')
    parser.add_argument('files', type=str,
                        help='plain text file containing full paths to files to be processed')
    parser.add_argument('outdir', type=str,
                        help='Path to output directory to save data in')
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite previous output files if they're in the way")
    parser.add_argument('--nprocesses', type=int, default=1,
                        help="Number of jobs to process in parallel")
    parser.add_argument('--verbose', type=str, default='INFO',
                        help="Set the logging level for OSL functions")
    parser.add_argument('--mneverbose', type=str, default='WARNING',
                        help="Set the logging level for MNE functions")

    parser.usage = parser.format_help()
    args = parser.parse_args(argv)

    run_proc_batch(**vars(args))


if __name__ == '__main__':

    main()
