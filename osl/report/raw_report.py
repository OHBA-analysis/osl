#!/usr/bin/env python3

# vim: set expandtab ts=4 sw=4:

import os
import mne
import sys
import yaml
import sails
import numpy as np
import argparse
from jinja2 import Template
from tabulate import tabulate
import matplotlib.pyplot as plt
import neurokit2 as nk

from ..utils import process_file_inputs
from ..preprocessing import import_data, check_inconfig, run_proc_chain


# ----------------------------------------------------------------------------------
# Report generation

def get_header_id(raw):
    """Extract scan name from MNE data object."""
    return raw.filenames[0].split('/')[-1].strip('.fif')


def gen_fif_data(raw, outf=None, fif_id=None, gen_plots=True):
    """Generate HTML web-report for an MNE data object."""
    x = {}
    x['filename'] = raw.filenames[0]
    x['fif_id'] = get_header_id(raw)
    print('Processing : {0}'.format(x['filename']))
    x['projname'] = raw.info['proj_name']
    x['experimenter'] = raw.info['experimenter']
    x['meas_date'] = raw.info['meas_date'].__str__()

    x['acq_samples'] = raw.n_times
    x['acq_sfreq'] = raw.info['sfreq']
    x['acq_duration'] = raw.n_times/raw.info['sfreq']

    x['nchans'] = raw.info['nchan']
    x['nhpi'] = len(raw.info['hpi_meas'][0]['hpi_coils'])

    from mne.channels.channels import channel_type
    chtype = [channel_type(raw.info, c) for c in range(x['nchans'])]
    chs, chcounts = np.unique(chtype, return_counts=True)
    x['chantable'] = tabulate(np.c_[chs, chcounts], tablefmt='html',
                              headers=['Channel Type', 'Number Acquired'])

    dig_codes = ('Cardinal', 'HPI', 'EEG', 'Extra')
    dig_counts = np.zeros((4,))
    for ii in range(1, 5):
        dig_counts[ii-1] = np.sum([d['kind'] == ii for d in raw.info['dig']])
    #d, dcounts = np.unique(digs, return_counts=True)
    x['digitable'] = tabulate(np.c_[dig_codes, dig_counts], tablefmt='html',
                              headers=['Digitisation Stage', 'Points Acquired'])

    ev = mne.find_events(raw, min_duration=5/raw.info['sfreq'], verbose=False)
    ev, evcounts = np.unique(ev[:, 2], return_counts=True)
    x['eventtable'] = tabulate(np.c_[ev, evcounts], tablefmt='html',
                               headers=['Event Code', 'Value'])

    savebase = '{0}/{1}'.format(outf, x['fif_id']) + '_{0}.png'
    if gen_plots:
        #plot_artefact_channels(raw, savebase=savebase)
        plot_eog_summary(raw, savebase=savebase)
        plot_ecg_summary_neurokit(raw, savebase=savebase)
        plot_digitisation_2d(raw, savebase=savebase)
        plot_spectra(raw, savebase=savebase)
        plot_channel_dists(raw, savebase=savebase)
    plt.close('all')

    x['plt_channeldev'] = savebase.format('channel_dev')
    x['plt_temporaldev'] = savebase.format('temporal_dev')
    x['plt_artefacts_eog'] = savebase.format('EOG')
    x['plt_artefacts_ecg'] = savebase.format('ECG')
    x['plt_digitisation'] = savebase.format('digitisation')
    x['plt_spectra'] = savebase.format('spectra_full')
    x['plt_zoom_spectra'] = savebase.format('spectra_zoom')

    return x


def gen_fif_html(raw, outf=None, fif_id=None, gen_plots=True):

    x = gen_fif_data(raw)

    # Replace the target string
    filedata = fif_report
    for key in x.keys():
        filedata = filedata.replace("@{0}".format(key), str(x[key]))

    return filedata


def gen_report(infiles, outdir=None, preproc_config=None):
    """Generate web-report for a set of MNE data objects."""

    infiles, outnames, good_files = process_file_inputs(infiles)

    if outdir is None:
        import tempfile
        tempdir = tempfile.TemporaryDirectory()
        outdir = tempdir.name

    if preproc_config is not None:
        config = check_inconfig(preproc_config)

    s = "<a href='#{0}'>{1}</a><br />"
    top_links = [s.format(outnames[ii], infiles[ii]) for ii in range(len(infiles))]
    top_links = '\n'.join(top_links)

    renders = []
    run_template = load_template('fif_base')

    for infile in infiles:
        raw = import_data(infile)
        if preproc_config is not None:
            raw = run_proc_chain(raw, config)['raw']
        data = gen_fif_data(raw, outf=outdir, fif_id='TEST', gen_plots=True)
        renders.append(run_template.render(run=data))

    if preproc_config is not None:
        preproc = '<div style="margin: 30px"><h3>preprocessing applied</h3>'
        for method in config['preproc']:
            preproc += "{0}</br>".format(method)
        #preproc = "<div style='margin: 30px'>{0}</div>".format(yaml.dump(config).__str__())
        preproc += '</div>'
    else:
        preproc = None

    page_template = load_template('raw_report_base')
    page = page_template.render(runs=renders, toplinks=top_links, preproc=preproc)

    # Write the file out again
    outpath = '{0}/osl_raw_report.html'.format(outdir)
    with open(outpath, 'w') as f:
        f.write(page)
    print(outpath)


def load_template(tname):
    basedir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(basedir, 'templates', '{0}.html.jinja2'.format(tname))
    with open(fname, 'r') as file_:
        template = Template(file_.read())

    return template


# ----------------------------------------------------------------------------------
# Scan stats and figures


def print_badsegs(raw):
    """Print a text-summary of the bad segments marked in a dataset."""
    durs = np.array([r['duration'] for r in raw.annotations])
    full_dur = raw.n_times/raw.info['sfreq']
    types = [r['description'] for r in raw.annotations]

    for modality in ['grad', 'mag', 'eeg']:
        inds = [s.find(modality) > 0 for s in types]
        mod_dur = np.sum(durs[inds])
        pc = (mod_dur / full_dur) * 100
        s = 'Modality {0} - {1:02f}/{2} seconds rejected     ({3:02f}%)'
        print(s.format(modality, mod_dur, full_dur, pc))


def plot_ecg_summary(raw, savebase):

    inds = mne.pick_types(raw.info, ecg=True)
    if len(inds) == 0:
        return 'ECG Channel Not Found'

    x = raw.get_data(inds)
    xinds = np.arange(0, raw.info['sfreq']*30).astype(int)

    plt.figure(figsize=(16, 5))
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.title('ECG')
    plt.subplot(211)
    plt.plot(raw.times, x.T)
    plt.subplot(212)
    plt.plot(raw.times[xinds], x[:, xinds].T)
    plt.xlabel('Time (seconds)')

    if savebase is not None:
        plt.savefig(savebase.format('ECG'), dpi=150, transparent=True)


def plot_ecg_summary_neurokit(raw, savebase):

    inds = mne.pick_types(raw.info, ecg=True)
    if len(inds) == 0:
        return 'ECG Channel Not Found'

    x = raw.get_data(inds)
    if np.abs(x.min()) > x.max():
        x = -x
    signals, info = nk.ecg_process(x[0, :], sampling_rate=raw.info['sfreq'])
    fig = nk.ecg_plot(signals, sampling_rate=raw.info['sfreq'])
    fig.set_size_inches(16,7)
    plt.subplots_adjust(left=0.075, right=0.95)

    fig.savefig(savebase.format('ECG'), dpi=150, transparent=True)


def plot_eog_summary(raw, savebase):

    inds = mne.pick_types(raw.info, eog=True)
    if len(inds) == 0:
        return 'EOG Channel Not Found'

    x = raw.get_data(inds)
    xinds = np.arange(0, raw.info['sfreq']*30).astype(int)

    plt.figure(figsize=(16, 5))
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.title('EOG')
    plt.subplot(211)
    plt.plot(raw.times, x.T)
    plt.subplot(212)
    plt.plot(raw.times[xinds], x[:, xinds].T)
    plt.xlabel('Time (seconds)')

    if savebase is not None:
        plt.savefig(savebase.format('EOG'), dpi=150, transparent=True)


def plot_eog_summary_neurokit(raw, savebase):

    inds = mne.pick_types(raw.info, eog=True)
    if len(inds) == 0:
        return 'ECG Channel Not Found'

    x = raw.get_data(inds)
    ind = np.argmax(x.max(axis=1))
    x = x[ind, :]
    signals, info = nk.eog_process(x, sampling_rate=raw.info['sfreq'])
    fig = nk.eog_plot(signals, sampling_rate=raw.info['sfreq'])
    fig.set_size_inches(16,7)
    plt.subplots_adjust(left=0.075, right=0.95)

    fig.savefig(savebase.format('ECG'), dpi=150, transparent=True)


def plot_artefact_channels(raw, savebase):
    """Plot ECG+EOG channels."""
    # ECG
    inds = mne.pick_channels(raw.ch_names, include=['EOG001', 'EOG002', 'ECG003'])
    dat = raw.get_data()[inds, :]

    plt.figure(figsize=(16, 10))
    plt.subplot(211)
    plt.plot(raw.times, dat[0, :])
    plt.plot(raw.times, dat[1, :])
    plt.title('EOG')
    plt.subplot(212)
    plt.plot(raw.times, dat[2, :])
    plt.title('ECG')
    plt.xlabel('Time (seconds)')
    if savebase is not None:
        plt.savefig(savebase.format('artefacts'), dpi=150, transparent=True)

    xinds = np.arange(0, raw.info['sfreq']*30).astype(int)
    plt.figure(figsize=(16, 10))
    plt.subplot(211)
    plt.plot(raw.times[xinds], dat[0, xinds])
    plt.plot(raw.times[xinds], dat[1, xinds])
    plt.title('EOG')
    plt.subplot(212)
    plt.plot(raw.times[xinds], dat[2, xinds])
    plt.title('ECG')
    plt.xlabel('Time (seconds)')
    if savebase is not None:
        plt.savefig(savebase.format('artefacts_zoom'), dpi=150, transparent=True)


def plot_spectra(raw, savebase=None):
    """Plot power spectra for each sensor modality."""
    fig = raw.plot_psd(show=False)
    fig.set_size_inches(8, 7)
    if savebase is not None:
        fig.savefig(savebase.format('spectra_full'), dpi=150, transparent=True)

    fig = raw.plot_psd(show=False, fmin=1, fmax=48)
    fig.set_size_inches(8, 7)
    if savebase is not None:
        fig.savefig(savebase.format('spectra_zoom'), dpi=150, transparent=True)


def plot_channel_dists(raw, savebase=None):
    """Plot summary distributions of sensors."""
    fig = plt.figure(figsize=(16, 2))
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.subplot(131)
    inds = mne.pick_types(raw.info, meg='mag')
    plt.hist(raw.get_data()[inds, :].std(axis=1), 24)
    plt.xlabel('St-Dev')
    plt.ylabel('Channel Count')
    plt.title('Magnetometers temporal std-dev')
    plt.subplot(132)
    inds = mne.pick_types(raw.info, meg='grad')
    plt.hist(raw.get_data()[inds, :].std(axis=1), 24)
    plt.title('Gradiometers temporal st-dev')
    plt.xlabel('St-Dev')
    plt.subplot(133)
    inds = mne.pick_types(raw.info, meg=False, eeg=True)
    plt.hist(raw.get_data()[inds, :].std(axis=1))
    plt.title('EEG temporal st-dev')
    plt.xlabel('St-Dev')
    if savebase is not None:
        fig.savefig(savebase.format('channel_dev'), dpi=150, transparent=True)

    fig = plt.figure(figsize=(16, 2))
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.subplot(131)
    inds = mne.pick_types(raw.info, meg='mag')
    plt.plot(raw.times, raw.get_data()[inds, :].std(axis=0))
    plt.title('Magnetometers channel st-dev')
    plt.ylabel('St-Dev over Channels')
    plt.title('Magnetometers channel std-dev')
    plt.xlabel('Time (seconds)')
    plt.subplot(132)
    inds = mne.pick_types(raw.info, meg='grad')
    plt.plot(raw.times, raw.get_data()[inds, :].std(axis=0))
    plt.title('Gradiometers channel st-dev')
    plt.xlabel('Time (seconds)')
    plt.subplot(133)
    inds = mne.pick_types(raw.info, meg=False, eeg=True)
    plt.plot(raw.times, raw.get_data()[inds, :].std(axis=0))
    plt.title('EEG channel st-dev')
    plt.xlabel('Time (seconds)')
    if savebase is not None:
        fig.savefig(savebase.format('temporal_dev'), dpi=150, transparent=True)


def plot_digitisation_2d(raw, savebase=None):

    fig = plt.figure(figsize=(16, 4))
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.subplot(141)
    plt.gca().set_aspect('equal')
    for dp in raw.info['dig']:
        if dp['kind'] == 1:
            plt.plot(dp['r'][0], dp['r'][1], 'r^')
        if dp['kind'] == 2:
            plt.plot(dp['r'][0], dp['r'][1], 'mo')
        if dp['kind'] == 3:
            plt.plot(dp['r'][0], dp['r'][1], 'g+')
        if dp['kind'] == 4:
            plt.plot(dp['r'][0], dp['r'][1], 'b.')
    plt.title('Top View')

    plt.subplot(142)
    plt.gca().set_aspect('equal')
    for dp in raw.info['dig']:
        if dp['kind'] == 1:
            plt.plot(dp['r'][0], dp['r'][2], 'r^')
        if dp['kind'] == 2:
            plt.plot(dp['r'][0], dp['r'][2], 'mo')
        if dp['kind'] == 3:
            plt.plot(dp['r'][0], dp['r'][2], 'g+')
        if dp['kind'] == 4:
            plt.plot(dp['r'][0], dp['r'][2], 'b.')
    plt.title('Front View')

    plt.subplot(143)
    plt.gca().set_aspect('equal')
    for dp in raw.info['dig']:
        if dp['kind'] == 1:
            plt.plot(dp['r'][1], dp['r'][2], 'r^')
        if dp['kind'] == 2:
            plt.plot(dp['r'][1], dp['r'][2], 'mo')
        if dp['kind'] == 3:
            plt.plot(dp['r'][1], dp['r'][2], 'g*')
        if dp['kind'] == 4:
            plt.plot(dp['r'][1], dp['r'][2], 'b.')
    plt.title('Side View')

    plt.subplot(144, frameon=False)
    plt.xticks([])
    plt.yticks([])
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='^', color='w', lw=4, label='Fiducial', markerfacecolor='r',markersize=14),
                       Line2D([0], [0], marker='o', color='w', label='HPI', markerfacecolor='m', markersize=14),
                       Line2D([0], [0], marker='*', color='w', label='EEG', markerfacecolor='g', markersize=14),
                       Line2D([0], [0], marker='.', color='w', label='Headshape', markerfacecolor='b', markersize=14)]
    plt.legend(handles=legend_elements, loc='center', frameon=False)

    if savebase is not None:
        fig.savefig(savebase.format('digitisation'), dpi=150, transparent=True)


def plot_headmovement(raw, savebase=None):
    """Plot headmovement - WORK IN PROGRESS... seems v-slow atm"""
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=False)
    fig = mne.viz.plot_head_positions(head_pos, mode='traces')

    if savebase is not None:
        fig.savefig(savebase.format('headpos'), dpi=150, transparent=True)


def print_scan_summary(raw):
    """Print a text summary of an MNE file."""
    print('Datafile : {0}'.format(raw.filenames))

    # Project name - set by user during acquisition
    print('Project name: {0}'.format(raw.info['proj_name']))
    print('Experimenter: {0}'.format(raw.info['experimenter']))

    date = raw.info['meas_date']
    print('Data acquired: {0}'.format(date))

    # Duration and sample rate of scan
    print('{0} samples at {1}Hz - {2} seconds'.format(raw.n_times,
                                                      raw.info['sfreq'],
                                                      raw.n_times/raw.info['sfreq']))

    # Number of Head Position Indicator coils
    nhpi = len(raw.info['hpi_meas'][0]['hpi_coils'])
    print('{0} HPI coils acquired'.format(nhpi))

    # Number of channels acquired
    nchans = raw.info['nchan']
    print('{0} channels acquired'.format(nchans))

    # Breakdown channels into channel types
    from mne.channels.channels import channel_type
    chtype = [channel_type(raw.info, c) for c in range(nchans)]
    chs, chcounts = np.unique(chtype, return_counts=True)

    for ii in range(len(chs)):
        print('\t{0}:{1}'.format(chs[ii], chcounts[ii]))

    # Head and coild digitisation points
    dig_codes = ('Cardinal', 'HPI', 'EEG', 'Extra')
    digs = [d['kind'] for d in raw.info['dig']]
    d, dcounts = np.unique(digs, return_counts=True)
    print('Digitisation points')
    for ii in range(len(d)):
        print('\t{0}:{1}'.format(dig_codes[ii], dcounts[ii]))

    # Trigger code events
    ev = mne.find_events(raw, min_duration=5/raw.info['sfreq'], verbose=False)
    ev, evcounts = np.unique(ev[:, 2], return_counts=True)
    print('Events')
    for ii in range(len(ev)):
        print('\t{0}:{1}'.format(ev[ii], evcounts[ii]))

    # Annotations - bad segments
    print_badsegs(raw)


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    print(argv)

    parser = argparse.ArgumentParser(description='Run a quick Quality Control summary on data.')
    parser.add_argument('files', type=str, nargs='+',
                        help='plain text file containing full paths to files to be processed')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Path to output directory to save data in')
    parser.add_argument('--config', type=str,
                        help='yaml defining preproc')

    args = parser.parse_args(argv)

    # -------------------------------------------

    gen_report(args.files, args.outdir)


# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
