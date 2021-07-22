#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

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

"""MNE Advice

* L-poass filter before downsampling!
 - Makes a huge speed difference

* Use notch_filter rather than the manual bandstop option in standard filter

* Don't downsample?
https://mne.tools/dev/auto_tutorials/preprocessing/30_filtering_resampling.html#resampling
... or at least downsample events with data (this is default here)

XML sanity check on config file. pre-parser to avoid mad sequences of methods.
"""

# --------------------------------------------------------------
# Preproc funcs from MNE


def import_data(infif):
    """Including as a function to make adding alt loaders easier later."""
    print('IMPORTING: {0}'.format(infif))
    return mne.io.read_raw_fif(infif, preload=True)


def run_mne_find_events(dataset, userargs):
    print('\nFINDING EVENTS')
    dataset['events'] = mne.find_events(dataset['raw'], **userargs)
    return dataset


def run_mne_filter(dataset, userargs):
    print('\nFILTERING')
    dataset['raw'].filter(**userargs)
    return dataset


def run_mne_notch_filter(dataset, userargs):
    print('\nNOTCH-FILTERING')
    freqs = np.array(userargs.pop('freqs').split(' ')).astype(float)
    print(freqs)
    dataset['raw'].notch_filter(freqs, **userargs)
    return dataset


def run_mne_resample(dataset, userargs):
    """The MNE guys don't seem to like resampling so much...
    """
    print('\nRESAMPLING')
    dataset['raw'].resample(**userargs)
    return dataset


def run_mne_epochs(dataset, userargs):
    print('\nEPOCHING')
    dataset['epochs'] = mne.Epochs(dataset['raw'],
                                   dataset['events'],
                                   dataset['event_id'])
    return dataset


def run_mne_crop(dataset, userargs):
    print('\nCROPPING')
    dataset['raw'].crop(**userargs)
    return dataset


def run_mne_ica_raw(dataset, userargs):
    print('\nICA')
    # NOTE: **userargs doesn't work because 'picks' is in there
    ica = mne.preprocessing.ICA(n_components=userargs['n_components'])
    ica.fit(dataset['raw'], picks=userargs['picks'])
    dataset['ica'] = ica
    return dataset


def run_mne_ica_autoreject(dataset, userargs):
    print('\nICA AUTOREJECT')
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
    print('Marking {0} as EOG ICs'.format(len(dataset['ica'].exclude)))
    ecg_indices, ecg_scores = dataset['ica'].find_bads_ecg(dataset['raw'],
                                                           threshold=ecgthreshold,
                                                           method=ecgmethod)
    dataset['ica'].exclude.extend(ecg_indices)
    print('Marking {0} as ECG ICs'.format(len(dataset['ica'].exclude)))
    if ('apply' not in userargs) or (userargs['apply'] is True):
        print('\nREMOVING SELECTED COMPONENTS FROM RAW DATA')
        dataset['ica'].apply(dataset['raw'])
    else:
        print('\nCOMPONENTS WERE NOT REMOVED FROM RAW DATA')
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


def get_badchan_labels(raw, userargs):
    """Set bad channels in MNE object."""
    print('\nBAD-CHANNELS')
    picks = userargs.get('picks', 'grad')
    print(picks)
    bdinds = sails.utils.detect_artefacts(raw.get_data(picks=picks), 0,
                                          reject_mode='dim',
                                          ret_mode='bad_inds')

    if (picks == 'mag') or (picks == 'grad'):
        chinds = mne.pick_types(raw.info, meg=picks)
    elif (picks == 'eeg'):
        chinds = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = np.array(raw.ch_names)[chinds]

    s = 'Modality {0} - {1}/{2} channels rejected     ({3:02f}%)'
    pc = (bdinds.sum() / len(bdinds)) * 100
    print(s.format(userargs['picks'], bdinds.sum(), len(bdinds), pc))

    if np.any(bdinds):
        return list(ch_names[np.where(bdinds)[0]])
    else:
        return []


def run_osl_bad_segments(dataset, userargs):
    print('\nBAD-SEGMENTS')
    #anns = dataset['raw'].annotations
    new = get_badseg_annotations(dataset['raw'], userargs)
    dataset['raw'].annotations.append(*new)

    mod_dur = new[1].sum()
    full_dur = dataset['raw'].n_times/dataset['raw'].info['sfreq']
    pc = (mod_dur / full_dur) * 100
    s = 'Modality {0} - {1:02f}/{2} seconds rejected     ({3:02f}%)'
    print(s.format(userargs['picks'], mod_dur, full_dur, pc))
    return dataset


def run_osl_bad_channels(dataset, userargs):
    badchans = get_badchan_labels(dataset['raw'], userargs)
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


def run_osl_ica(dataset, userargs):
    print('\nICA')
    return dataset


# --------------------------------------------------------------
# Utils

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
        with open(config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    elif isinstance(config, dict):
        pass

    return config


def check_infifs(infiles):
    infifs = []
    outnames = []
    if isinstance(infiles, str):
        # We have a file with a list of paths
        check_paths = True
        for row in csv.reader(open(infiles, 'r'), delimiter=" "):
            infifs.append(row[0])
            if len(row) > 1:
                outnames.append(row[1])
    elif isinstance(infiles[0], str):
        # We have a list of paths
        check_paths = True
        infifs = infiles
    elif isinstance(infiles[0], [list, tuple]):
        # We have a list containing files and output names
        check_paths = True
        for row in infiles:
            infifs.append(row[0])
            outnames.append(row[1])
    elif isinstance(infiles[0], mne.io.fiff.raw.Raw):
        # We have a list of MNE objects
        check_paths = False
        infifs = infiles

    if len(outnames) == 0:
        outnames = None

    # Check that files actually exist if we've been passed filenames rather
    # than objects
    good_fifs = [1 for ii in range(len(infifs))]
    if check_paths:
        for idx, fif in enumerate(infifs):
            if os.path.isfile(fif) == False:
                good_fifs[idx] = 0
                print('File not found: {0}'.format(fif))

    return infifs, outnames, good_fifs


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


def run_proc_chain(infif, config, outname=None, outdir=None, ret_dataset=False, overwrite=False):

    if outname is None:
        run_id = os.path.split(infif)[1].rstrip('.fif')
    else:
        run_id = outname.rstrip('.fif')

    if outdir is not None:
        name_base = '{run_id}_{ftype}.{fext}'
        outbase = os.path.join(outdir, name_base)
        logfile = outbase.format(run_id=run_id, ftype='preproc', fext='log')
        mne.utils._logging.set_log_file(logfile)

    if isinstance(infif, str):
        raw = import_data(infif)
    elif isinstance(infif, mne.io.fiff.raw.Raw):
        raw = infif
        infif = raw.filenames[0]  # assuming only one file here

    dataset = {'raw': raw,
               'ica': None,
               'epochs': None,
               'events': None,
               'event_id': config['meta']['event_codes']}

    for stage in deepcopy(config['preproc']):
        method = stage.pop('method')
        func = find_func(method)
        try:
            dataset = func(dataset, stage)
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

    infifs, outnames, good_fifs = check_infifs(files)
    print('Processing {0} files'.format(sum(good_fifs)))

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
    for idx, infif in enumerate(infifs):
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