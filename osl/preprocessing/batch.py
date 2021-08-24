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

import os
import sys
import mne
import csv
import yaml
import sails
import argparse
import traceback
import numpy as np
from copy import deepcopy
import multiprocessing as mp
from functools import partial
from time import strftime, localtime


# --------------------------------------------------------------
# Preproc funcs from MNE


def import_data2(infile, preload=True):
    """Including as a function to make adding alt loaders easier later."""
    osl_print('IMPORTING: {0}'.format(infile))
    return mne.io.read_raw(infile, preload=preload)


def find_run_id(infile, preload=True):

    if isinstance(infile, mne.io.fiff.raw.Raw):
        infile = infile.filenames[0]

    if os.path.split(infile)[1] == 'c,rfDC':
        # We have a BTI scan
        runname = os.path.basename(os.path.dirname(infile))
    elif os.path.splitext(infile)[1] == '.fif':
        # We have a FIF file
        runname = os.path.basename(infile).rstrip('.fif')
    elif os.path.splitext(infile)[1] == '.meg4':
        # We have the meg file from a ds directory
        runname = os.path.basename(infile).rstrip('.ds')
    elif os.path.splitext(infile)[1] == '.ds':
        runname = os.path.basename(infile).rstrip('.ds')
    else:
        raise ValueError('Unable to determine run_id from file {0}'.format(infile))

    return runname


def import_data(infile, preload=True, logfile=None):
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
    else:
        raise ValueError('Unable to determine file type of input {0}'.format(infile))

    return raw


def run_mne_set_channel_types(dataset, userargs, logfile=None):
    osl_print('\nSETTING CHANNEL TYPES', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)

    dataset['raw'].set_channel_types(userargs)
    return dataset


def run_mne_pick_types(dataset, userargs, logfile=None):
    osl_print('\nPICKING CHANNEL TYPES', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)

    dataset['raw'].pick_types(**userargs)
    return dataset


def run_mne_find_events(dataset, userargs, logfile=None):
    osl_print('\nFINDING EVENTS', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    dataset['events'] = mne.find_events(dataset['raw'], **userargs)
    return dataset


def run_mne_filter(dataset, userargs, logfile=None):
    osl_print('\nFILTERING', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    dataset['raw'].filter(**userargs)
    return dataset


def run_mne_notch_filter(dataset, userargs, logfile=None):
    osl_print('\nNOTCH-FILTERING', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    freqs = np.array(userargs.pop('freqs').split(' ')).astype(float)
    osl_print(freqs)
    dataset['raw'].notch_filter(freqs, **userargs)
    return dataset


def run_mne_resample(dataset, userargs, logfile=None):
    """The MNE guys don't seem to like resampling so much...
    """
    osl_print('\nRESAMPLING', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    if ('events' in dataset) and (dataset['events'] is not None):
        dataset['raw'], dataset['events'] = dataset['raw'].resample(events=dataset['events'], **userargs)
    else:
        dataset['raw'].resample(**userargs)
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


def run_mne_crop(dataset, userargs, logfile=None):
    osl_print('\nCROPPING', logfile=logfile)
    dataset['raw'].crop(**userargs)
    return dataset


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


def run_mne_tfr_multitaper(dataset, userargs, logfile=None):
    osl_print('\nTFR MULTITAPER', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    from mne.time_frequency import tfr_multitaper

    freqs = np.array(userargs.pop('freqs').split(' ')).astype('float')
    freqs = np.linspace(freqs[0], freqs[1], int(freqs[2]))
    out = tfr_multitaper(dataset['epochs'],
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
    from mne.time_frequency import tfr_morlet

    freqs = np.array(userargs.pop('freqs').split(' ')).astype('float')
    print(freqs)
    freqs = np.linspace(freqs[0], freqs[1], int(freqs[2]))
    dataset['power'], dataset['itc'] = tfr_morlet(dataset['epochs'],
                                                  freqs,
                                                  **userargs)

    return dataset


def run_mne_tfr_stockwell(dataset, userargs, logfile=None):
    osl_print('\nTFR STOCKWELL', logfile=logfile)
    osl_print(str(userargs), logfile=logfile)
    from mne.time_frequency import tfr_stockwell

    out = tfr_stockwell(dataset['epochs'], **userargs)
    if 'return_itc' in userargs and userargs['return_itc']:
        dataset['power'], dataset['itc'] = out
    else:
        dataset['power'] = out

    return dataset


# --------------------------------------------------------------
# Preproc funcs from OHBA


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
    osl_print(picks, logfile=logfile)
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
# Utils

def osl_print(s, logfile=None):
    print(s)
    if logfile is not None:
        with open(logfile, 'a') as f:
            f.write(s + '\n')


def find_func(method):

    # Find MNE function in local module
    func = globals().get('run_mne_{0}'.format(method))

    if func is None:
        # Find OSL function in local module
        func = globals().get('run_osl_{0}'.format(method))

    if func is None:
        print('Func not found! {0}'.format(method))

    return func


def check_inconfig(config):
    if isinstance(config, str):
        try:
            # See if we have a filepath
            with open(config, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except (UnicodeDecodeError, FileNotFoundError):
            # We have a string
            config = yaml.load(config, Loader=yaml.FullLoader)
    elif isinstance(config, dict):
        # Its already a dict
        pass

    return config


def check_infiles(infiles):
    checked_files = []
    outnames = []
    if isinstance(infiles, str):
        # We have a file with a list of paths possibly with output names
        check_paths = True
        for row in csv.reader(open(infiles, 'r'), delimiter=" "):
            checked_files.append(row[0])
            if len(row) > 1:
                outnames.append(row[1])
            else:
                outnames.append(find_run_id(row[0]))
    elif isinstance(infiles[0], str):
        # We have a list of paths
        check_paths = True
        checked_files = infiles
        outnames = [find_run_id(f) for f in infiles]
    elif isinstance(infiles[0], (list, tuple)):
        # We have a list containing files and output names
        check_paths = True
        for row in infiles:
            checked_files.append(row[0])
            outnames.append(row[1])
    elif isinstance(infiles[0], mne.io.fiff.raw.Raw):
        # We have a list of MNE objects
        check_paths = False
        checked_files = infiles

    if len(outnames) == 0:
        outnames = None

    # Check that files actually exist if we've been passed filenames rather
    # than objects
    good_files = [1 for ii in range(len(checked_files))]
    if check_paths:
        for idx, fif in enumerate(checked_files):
            if fif.endswith('.ds'):
                good_files[idx] = int(os.path.isdir(fif))
            else:
                good_files[idx] = int(os.path.isfile(fif))
                print('File not found: {0}'.format(fif))

    return checked_files, outnames, good_files


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


def run_proc_chain(infile, config, outname=None, outdir=None, ret_dataset=True, overwrite=False):

    if outname is None:
        #run_id = os.path.split(infile)[1].rstrip('.fif')
        run_id = find_run_id(infile)
    else:
        run_id = outname.rstrip('.fif')

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
        func = find_func(method)
        try:
            dataset = func(dataset, stage, logfile=logfile)
        except Exception as e:
            print('PROCESSING FAILED!!!!')
            ex_type, ex_value, ex_traceback = sys.exc_info()
            print(ex_type)
            print(ex_value)
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

    if os.path.isdir(outdir) is False:
        raise ValueError('Output dir not found!')
    else:
        name_base = '{run_id}_{ftype}.{fext}'
        outbase = os.path.join(outdir, name_base)
        print(outbase)

    print('Outputs saving to: {0}\n\n'.format(outdir))
    config = check_inconfig(config)
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


# ----------------------------------------------------------
# Main user function


if __name__ == '__main__':

    main()
