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

from ..utils import process_file_inputs, validate_outdir
from ..preprocessing import import_data, check_inconfig, run_proc_chain


# ----------------------------------------------------------------------------------
# Report generation

def get_header_id(raw):
    """Extract scan name from MNE data object."""
    return raw.filenames[0].split('/')[-1].strip('.fif')


def gen_fif_data(raw, ica=None, outf=None, fif_id=None, gen_plots=True, artefact_scan=False):
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
        plot_bad_ica(raw, ica, savebase=savebase)
        plot_digitisation_2d(raw, savebase=savebase)
        plot_spectra(raw, savebase=savebase)
        plot_channel_dists(raw, savebase=savebase)
        plot_channel_sumsq_timecourse(raw, savebase=savebase)
        if artefact_scan:
            plot_artefact_scan(raw, savebase=savebase)

    plt.close('all')

    # HTML can use the relative paths
    savebase2 = os.path.split(savebase)[1]

    x['plt_channeldev'] = savebase2.format('channel_dev')
    x['plt_temporaldev'] = savebase2.format('temporal_sumsq')
    x['plt_artefacts_eog'] = savebase2.format('EOG')
    x['plt_artefacts_ecg'] = savebase2.format('ECG')
    x['plt_artefacts_ica'] = savebase2.format('ICA')
    x['plt_digitisation'] = savebase2.format('digitisation')
    x['plt_spectra'] = savebase2.format('spectra_full')
    x['plt_zoom_spectra'] = savebase2.format('spectra_zoom')

    if artefact_scan:
        x['artefact_scan'] = True
        x['plt_eyemove_grad'] = savebase2.format('eyemove_grad')
        x['plt_eyemove_mag'] = savebase2.format('eyemove_mag')
        x['plt_eyemove_eog'] = savebase2.format('eyemove_eog')
        x['plt_blink_grad'] = savebase2.format('blink_grad')
        x['plt_blink_mag'] = savebase2.format('blink_mag')
        x['plt_blink_eog'] = savebase2.format('blink_eog')
        x['plt_swallow_grad'] = savebase2.format('swallow_grad')
        x['plt_swallow_mag'] = savebase2.format('swallow_mag')
        x['plt_breathe_grad'] = savebase2.format('breathe_grad')
        x['plt_breathe_mag'] = savebase2.format('breathe_mag')
        x['plt_shrug_grad'] = savebase2.format('shrug_grad')
        x['plt_shrug_mag'] = savebase2.format('shrug_mag')
        x['plt_clench_grad'] = savebase2.format('clench_grad')
        x['plt_clench_mag'] = savebase2.format('clench_mag')
        x['plt_buttonpress_grad'] = savebase2.format('buttonpress_grad')
        x['plt_buttonpress_mag'] = savebase2.format('buttonpress_mag')
    else:
        x['artefact_scan'] = False

    return x


def gen_fif_html(raw, outf=None, fif_id=None, gen_plots=True):

    x = gen_fif_data(raw)

    # Replace the target string
    filedata = fif_report
    for key in x.keys():
        filedata = filedata.replace("@{0}".format(key), str(x[key]))

    return filedata


def gen_report(infiles, outdir=None, preproc_config=None, artefact_scan=False):
    """Generate web-report for a set of MNE data objects."""
    infiles, outnames, good_files = process_file_inputs(infiles)

    if outdir is None:
        import tempfile
        tempdir = tempfile.TemporaryDirectory()
        outdir = tempdir.name
    else:
        outdir = validate_outdir(outdir)

    if preproc_config is not None:
        config = check_inconfig(preproc_config)

    s = "<a href='#{0}'>{1}</a><br />"
    top_links = [s.format(outnames[ii], infiles[ii]) for ii in range(len(infiles))]
    top_links = '\n'.join(top_links)

    renders = []
    run_template = load_template('fif_base_tabs')

    for infile in infiles:
        raw = import_data(infile)
        if os.path.exists(infile.replace('raw.fif', 'ica.fif')):
            ica = mne.preprocessing.read_ica(infile.replace('raw.fif', 'ica.fif')) # todo: could potentially be incorporated in 'import_data'
        else:
            ica = None
        if preproc_config is not None:
            dataset = run_proc_chain(raw, config)
            raw = dataset['raw']
            ica = dataset['ica']

        data = gen_fif_data(raw, ica=ica, outf=outdir, fif_id='TEST',
                            gen_plots=True, artefact_scan=artefact_scan)
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

def plot_bad_ica(raw, ica, savebase):
    """Plot ICA characteristics for rejected components."""
    exclude_uniq = np.sort(np.unique(ica.exclude))
    nbad = len(exclude_uniq)
    figsize = [16., 5*nbad]
    fig = plt.figure(figsize=figsize, facecolor=[0.95] * 3)
    axes = []
    for i in np.arange(nbad):
        lowerlimit = 0.1+i/(nbad*1.1)
        multiplier=nbad*1.3
        # adapted from mne/viz/ica._create_properties_layout
        axes_params = (('topomap', [0.08, lowerlimit+0.5/multiplier, 0.3, 0.45/multiplier]),
                   ('image', [0.5, lowerlimit+0.6/multiplier, 0.45, 0.35/multiplier]),
                   ('erp', [0.5, lowerlimit+0.5/multiplier, 0.45, 0.1/multiplier]),
                   ('spectrum', [0.08, lowerlimit+0.1/multiplier, 0.32, 0.3/multiplier]),
                   ('variance', [0.5, lowerlimit+0.025/multiplier, 0.45, 0.25/multiplier]))
        axes += [[fig.add_axes(loc, label=name) for name, loc in axes_params]]
        ica.plot_properties(raw, picks=exclude_uniq[i], axes=axes[i])
        if np.any([x in ica.labels_.keys() for x in ica._ica_names]): # this is for the osl_plot_ica convention
            title = "".join((ica._ica_names[exclude_uniq[i]]," - ", ica.labels_[ica._ica_names[exclude_uniq[i]]].upper()))
        elif np.logical_or('eog' in ica.labels_.keys(), 'ecg' in ica.labels_.keys()): # this is for the MNE automatic labelling convention
            flag_eog = exclude_uniq[i] in ica.labels_['eog']
            flag_ecg = exclude_uniq[i] in ica.labels_['ecg']
            title = "".join((ica._ica_names[exclude_uniq[i]]," - ", flag_eog*'EOG', flag_ecg*flag_eog*'/', flag_ecg*'ECG'))
        else: # this is for if there is nothing in ica.labels_
            title = None
        if title is not None:
            axes[i][0].set_title(title, fontsize=12)
    if savebase is not None:
        plt.savefig(savebase.format('ica'), dpi=150, transparent=True)

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


def plot_channel_sumsq_timecourse(raw, savebase=None):
    from scipy.ndimage.filters import uniform_filter1d

    fig = plt.figure(figsize=(16, 4))
    plt.subplots_adjust(hspace=0.3,left=0.05, right=0.95)
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.subplot(311)
    inds = mne.pick_types(raw.info, meg='mag')
    x = np.sum(raw.get_data()[inds, :]**2, axis=0)
    x = uniform_filter1d(x, int(raw.info['sfreq']))
    plt.plot(raw.times, x)
    plt.fill_between(raw.times, x, alpha=0.5)
    plt.title('Sum-Square across channels')
    plt.legend(['Magnetometers'], frameon=False)
    plt.gca().set_xticklabels([])
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.subplot(312)
    inds = mne.pick_types(raw.info, meg='grad')
    x = np.sum(raw.get_data()[inds, :]**2, axis=0)
    x = uniform_filter1d(x, int(raw.info['sfreq']))
    plt.plot(raw.times, x)
    plt.fill_between(raw.times, x, alpha=0.5)
    plt.legend(['Gradiometers'], frameon=False)
    plt.gca().set_xticklabels([])
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.subplot(313)
    inds = mne.pick_types(raw.info, meg=False, eeg=True)
    x = np.sum(raw.get_data()[inds, :]**2, axis=0)
    x = uniform_filter1d(x, int(raw.info['sfreq']))
    plt.plot(raw.times, x)
    plt.fill_between(raw.times, x, alpha=0.5)
    plt.legend(['EEG'], frameon=False)
    plt.xlabel('Time (seconds)')
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    if savebase is not None:
        fig.savefig(savebase.format('temporal_sumsq'), dpi=150, transparent=True)


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


def plot_artefact_scan(raw, savebase=None):

    events = mne.find_events(raw, min_duration=2/raw.info['sfreq'])
    modalities = ['mag', 'grad']

    # Plot eye movements
    event_dict = {'moves': 1}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-1, tmax=16.5,
                        preload=True)
    x = epochs.get_data()

    yl = np.abs(x[:,:2,:]).max() * 1.1

    plt.figure(figsize=(16,5))
    ax = plt.subplot(121)
    from matplotlib.patches import Rectangle
    patches = [Rectangle((-1,-yl),1,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((0,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((2,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((4,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((6,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((8,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((10,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((12,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((14,-yl),2,yl*2, alpha=0.2, facecolor='grey')]
    for rec in patches:
        ax.add_artist(rec)
    plt.text(-0.5, yl, 'fix', ha='center')
    plt.text(1, yl, 'Up', ha='center')
    plt.text(5, yl, 'Right', ha='center')
    plt.text(9, yl, 'Down', ha='center')
    plt.text(13, yl, 'Left', ha='center')
    plt.plot(epochs.times, x[:,0,:].T)
    plt.ylim(-yl, yl*1.1)
    ax = plt.subplot(122)
    patches = [Rectangle((-1,-yl),1,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((0,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((2,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((4,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((6,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((8,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((10,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((12,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((14,-yl),2,yl*2, alpha=0.2, facecolor='grey')]
    for rec in patches:
        ax.add_artist(rec)
    plt.text(-0.5, yl, 'fix', ha='center')
    plt.text(1, yl, 'Up', ha='center')
    plt.text(5, yl, 'Right', ha='center')
    plt.text(9, yl, 'Down', ha='center')
    plt.text(13, yl, 'Left', ha='center')
    plt.plot(epochs.times, x[:,1,:].T)
    plt.ylim(-yl, yl*1.1)
    plt.savefig(savebase.format('eyemove_eog'), dpi=300, transparent=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'eyemove_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')

    # Plot blinks
    event_dict = {'blinks': 5}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=2,
                        preload=True)
    x = epochs.get_data()

    plt.figure(figsize=(16,4))
    plt.subplot(121)
    plt.plot(epochs.times, x[:,0,:].T)
    plt.title(raw.info['ch_names'][0])
    plt.subplot(122)
    plt.plot(epochs.times, x[:,1,:].T)
    plt.title(raw.info['ch_names'][1])
    name = 'blink_eog'.format(m)
    plt.savefig(savebase.format(name), dpi=300, transparent=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'blink_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')

    # Plot swallow
    event_dict = {'swallow': 6}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=2,
                        preload=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'swallow_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')

    # Plot breathe
    event_dict = {'breath': 7}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.5, tmax=5,
                        preload=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'breathe_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')

    # Plot shrug
    event_dict = {'shrug': 8}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.5, tmax=3,
                        preload=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'shrug_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')

    # Plot clench
    if len(np.where(events[:,2]==9)[0]) > 0:
        event_dict = {'clench': 9}
        epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.5, tmax=3,
                            preload=True)

        ev = epochs.average()
        for m in modalities:
            fig = ev.plot_joint(show=False, picks=m)
            name = 'clench_{0}'.format(m)
            fig.savefig(savebase.format(name), dpi=300, transparent=True)
        plt.close('all')

    # Plot button press
    event_dict = {'button_press': 257}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.5, tmax=2,
                        preload=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'buttonpress_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')


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
    parser.add_argument('--artefactscan', action="store_true",
                        help='Generate additional plots assuming inputs are artefact scans')

    args = parser.parse_args(argv)

    # -------------------------------------------

    gen_report(args.files, args.outdir, preproc_config=args.config,  artefact_scan=args.artefactscan)


# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
