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
import os
import sys
import traceback
from copy import deepcopy
from functools import partial
from time import localtime, strftime

import mne
import numpy as np
import sails
import yaml

from ..utils import find_run_id, validate_outdir, process_file_inputs, osl_print
from . import _mne_wrappers

# --------------------------------------------------------------
# Data importers


def import_data(infile, preload=True, logfile=None):
    if not isinstance(infile, str):
        raise ValueError(
            "infile must be a str. Got type(infile)={0}.".format(type(infile))
        )

    osl_print('IMPORTING: {0}'.format(infile), logfile=logfile)

    if os.path.split(infile)[1] == 'c,rfDC':
        # We have a BTI scan
        if os.path.isfile(os.path.join(os.path.split(infile)[0], 'hs_file')):
            head_shape_fname = 'hs_file'
        else:
            head_shape_fname = None
        raw = mne.io.read_raw_bti(infile, head_shape_fname=head_shape_fname, preload=preload)
    elif os.path.splitext(infile)[1] == '.fif':
        # We have a FIF file
        raw = mne.io.read_raw_fif(infile, preload=preload)
    elif os.path.splitext(infile)[1] == '.meg4':
        # We have the meg file from a ds directory
        raw = mne.io.read_raw_ctf(os.path.dirname(infile), preload=preload)
    elif os.path.splitext(infile)[1] == '.ds':
        raw = mne.io.read_raw_ctf(infile, preload=preload)
    elif os.path.splitext(infile)[1] == '.vhdr':
        raw = mne.io.read_raw_brainvision(infile, preload=preload)
    else:
        raise ValueError('Unable to determine file type of input {0}'.format(infile))

    return raw


# --------------------------------------------------------------
# OHBA Preprocessing functions
#
# TODO - probably wants moving to its own module similar to _mne_wrappers (with
# ICA functions?)


def get_badseg_annotations(raw, userargs):
    """Set bad segments in MNE object."""
    segment_len = userargs.get('segment_len', raw.info['sfreq'])
    picks = userargs.get('picks', 'grad')
    bdinds = sails.utils.detect_artefacts(raw.get_data(picks=picks), 1,
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

    onsets = onsets / raw.info['sfreq']
    durations = durations / raw.info['sfreq']

    #orig_time = [None for ii in range(len(onsets))]

    return onsets, durations, descriptions


def get_badchan_labels(raw, userargs, logfile=None):
    """Set bad channels in MNE object."""
    osl_print('\nBAD-CHANNELS', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    picks = userargs.get('picks', 'grad')
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
    osl_print(s.format(userargs['picks'], bdinds.sum(), len(bdinds), pc), logfile=logfile)

    if np.any(bdinds):
        return list(ch_names[np.where(bdinds)[0]])
    else:
        return []


def run_osl_bad_segments(dataset, userargs, logfile=None):
    osl_print('\nBAD-SEGMENTS', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    #anns = dataset['raw'].annotations
    new = get_badseg_annotations(dataset['raw'], userargs)
    dataset['raw'].annotations.append(*new)

    mod_dur = new[1].sum()
    full_dur = dataset['raw'].n_times/dataset['raw'].info['sfreq']
    pc = (mod_dur / full_dur) * 100
    s = 'Modality {0} - {1:02f}/{2} seconds rejected     ({3:02f}%)'
    osl_print(s.format(userargs['picks'], mod_dur, full_dur, pc), logfile=logfile)
    return dataset


def run_osl_bad_channels(dataset, userargs, logfile=None):
    badchans = get_badchan_labels(dataset['raw'], userargs, logfile=logfile)
    dataset['raw'].info['bads'].extend(badchans)
    return dataset


def _print_badsegs(raw, modality):
    """CURRENTLY BROKEN : Print a text-summary of the bad segments marked in a
    dataset."""
    types = [r['description'] for r in raw.annotations]

    inds = [s.find(modality) > 0 for s in types]
    mod_dur = np.sum(durs[inds])
    pc = (mod_dur / full_dur) * 100
    s = 'Modality {0} - {1:02f}/{2} seconds rejected     ({3:02f}%)'
    print(s.format(modality, mod_dur, full_dur, pc))


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
        print('Func not found! {0}'.format(method))

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

    for step in config['preproc']:
        if config['meta']['event_codes'] is None and 'find_events' in step.values():
            raise KeyError(
                'event_codes must be passed in config if we are finding events.'
            )

    return config


def write_dataset(dataset, outbase, run_id, overwrite=False):
    # Save output
    outname = outbase.format(run_id=run_id, ftype='raw', fext='fif')
    dataset['raw'].save(outname, overwrite=overwrite)

    if dataset['events'] is not None:
        outname = outbase.format(run_id=run_id, ftype='events', fext='npy')
        np.save(outname, dataset['events'])

    if dataset['epochs'] is not None:
        outname = outbase.format(run_id=run_id, ftype='epochs', fext='fif')
        dataset['epochs'].save(outname, overwrite=overwrite)

    if dataset['ica'] is not None:
        outname = outbase.format(run_id=run_id, ftype='ica', fext='fif')
        dataset['ica'].save(outname)

# --------------------------------------------------------------
# Bach processing

def run_proc_chain(infile, config, outname=None, outdir=None, ret_dataset=True, overwrite=False, extra_funcs=None):

    if outname is None:
        #run_id = os.path.split(infile)[1].rstrip('.fif')
        run_id = find_run_id(infile)
    else:
        run_id = os.path.splitext(outname)[0]

    if not isinstance(config, dict):
        config = load_config(config)

    if outdir is not None:
        name_base = '{run_id}_{ftype}.{fext}'
        outbase = os.path.join(outdir, name_base)
        logfile = outbase.format(run_id=run_id, ftype='preproc', fext='log')
        mne.utils._logging.set_log_file(logfile)
    else:
        logfile = None
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    osl_print('{0} : Starting OSL Processing'.format(now), logfile=logfile)
    osl_print('input : {0}'.format(infile), logfile=logfile)

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
        method = stage.pop('method')
        target = stage.get('target', 'raw')  # Raw is default
        func = find_func(method, target=target, extra_funcs=extra_funcs)
        try:
            dataset = func(dataset, stage, logfile=logfile)
        except Exception as e:
            print('PROCESSING FAILED!!!!')
            ex_type, ex_value, ex_traceback = sys.exc_info()
            print("{0} : {1}".format(method, func))
            print(ex_type)
            print(ex_value)
            print(traceback.print_tb(ex_traceback))
            if outdir is not None:
                with open(logfile.replace('.log', '.error.log'), 'w') as f:
                    f.write('Processing filed during stage : "{0}"\n\n'.format(method))
                    f.write(str(ex_type))
                    f.write('\n')
                    f.write(str(ex_value))
                    f.write('\n')
                    traceback.print_tb(ex_traceback, file=f)
            return 0

    if outdir is not None:
        write_dataset(dataset, outbase, run_id, overwrite=overwrite)

    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    osl_print('{0} : Processing Complete'.format(now), logfile=logfile)

    if ret_dataset:
        return dataset
    else:
        return 1


def run_proc_batch(config, files, outdir, overwrite=False, nprocesses=1, mnelog='INFO'):
    """
    files can be a list of Raw objects or a list of filenames or a path to a
    textfile list of filenames

    Currently writes outputs to disk - does not return a list of processed
    data! this could potentially be huge amounts of memory.

    """

    # -------------------------------------------------------------
    mne.set_log_level(mnelog)

    print('\n\nOHBA Auto-Proc\n\n')

    infiles, outnames, good_files = check_infiles(files)
    print('Processing {0} files'.format(sum(good_files)))

    outdir = validate_outdir(outdir)

    name_base = '{run_id}_{ftype}.{fext}'
    outbase = outdir / name_base

    print('Outputs saving to: {0}\n\n'.format(outdir))
    config = load_config(config)
    print(yaml.dump(config))

    # -------------------------------------------------------------
    # Create partial function with fixed options
    pool_func = partial(run_proc_chain,
                        outdir=outdir,
                        ret_dataset=False,
                        overwrite=overwrite)

    # For each file...
    args = []
    for idx, infif in enumerate(infiles):
        if outnames is None:
            outname = None
        else:
            outname = outnames[idx]

        args.append((infif, config, outname))

    # Actually run the processes
    p = mp.Pool(processes=nprocesses)
    proc_flags = p.starmap(pool_func, args)
    p.close()

    print('Processed {0}/{1} files successfully'.format(np.sum(proc_flags), len(proc_flags)))

    # Return failed flags
    return proc_flags


# ----------------------------------------------------------
# Main CLI user function


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    print(argv)

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
    parser.add_argument('--mnelog', type=str, default='INFO',
                        help="Set the logging level for MNE python functions")

    parser.usage = parser.format_help()
    args = parser.parse_args(argv)
    print(args)

    run_proc_batch(**vars(args))


if __name__ == '__main__':

    main()
