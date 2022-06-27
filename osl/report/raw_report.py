#!/usr/bin/env python3

# vim: set expandtab ts=4 sw=4:

import os
import mne
import sys
import yaml
import sails
import argparse
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

from jinja2 import Template
from tabulate import tabulate
from mne.channels.channels import channel_type
from scipy.ndimage.filters import uniform_filter1d
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from ..utils import process_file_inputs, validate_outdir
from ..preprocessing import import_data, load_config, run_proc_chain, get_config_from_fif, plot_preproc_flowchart    


# ----------------------------------------------------------------------------------
# Report generation

def get_header_id(raw):
    """Extract scan name from MNE data object."""
    return raw.filenames[0].split('/')[-1].strip('.fif')

def gen_html_data(raw, ica, outdir, level):
    """Generate HTML web-report for an MNE data object."""

    data = {}
    data['filename'] = raw.filenames[0]
    data['fif_id'] = get_header_id(raw)

    print('Processing : {0}'.format(data['filename']))

    # Level of reporting
    data['level'] = level

    # Scan info
    data['projname'] = raw.info['proj_name']
    data['edataperimenter'] = raw.info['experimenter']
    data['meas_date'] = raw.info['meas_date'].__str__()

    data['acq_samples'] = raw.n_times
    data['acq_sfreq'] = raw.info['sfreq']
    data['acq_duration'] = raw.n_times/raw.info['sfreq']

    # Channels/coils
    data['nchans'] = raw.info['nchan']

    try:
        data['nhpi'] = len(raw.info['hpi_meas'][0]['hpi_coils'])
    except:
        print("No HPI info in fif file")

    chtype = [channel_type(raw.info, c) for c in range(data['nchans'])]
    chs, chcounts = np.unique(chtype, return_counts=True)
    data['chantable'] = tabulate(np.c_[chs, chcounts], tablefmt='html',
                                 headers=['Channel Type', 'Number Acquired'])

    # Head digitisation
    dig_codes = ('Cardinal', 'HPI', 'EEG', 'Extra')
    dig_counts = np.zeros((4,))
    for ii in range(1, 5):
        dig_counts[ii-1] = np.sum([d['kind'] == ii for d in raw.info['dig']])
    #d, dcounts = np.unique(digs, return_counts=True)
    data['digitable'] = tabulate(np.c_[dig_codes, dig_counts], tablefmt='html',
                                 headers=['Digitisation Stage', 'Points Acquired'])

    # Events
    ev = mne.find_events(raw, min_duration=5/raw.info['sfreq'], verbose=False)
    ev, evcounts = np.unique(ev[:, 2], return_counts=True)
    data['eventtable'] = tabulate(np.c_[ev, evcounts], tablefmt='html',
                                  headers=['Event Code', 'Value'])

    # Bad segments
    durs = np.array([r['duration'] for r in raw.annotations])
    full_dur = raw.n_times/raw.info['sfreq']
    types = [r['description'] for r in raw.annotations]

    data['bad_seg'] = []
    for modality in ['grad', 'mag', 'eeg']:
        inds = [s.find(modality) > 0 for s in types]
        mod_dur = np.sum(durs[inds])
        pc = (mod_dur / full_dur) * 100
        s = 'Modality {0} - {1:.2f}/{2} seconds rejected     ({3:.2f}%)'
        if mod_dur > 0:
            data['bad_seg'].append(s.format(modality, mod_dur, full_dur, pc))

    # Bad channels
    bad_chans = raw.info['bads']
    if len(bad_chans) == 0:
        data['bad_chans'] = 'No bad channels.'
    else:
        data['bad_chans'] = 'Bad channels: {}'.format(', '.join(bad_chans))

    # Path to save plots
    savebase = '{0}/{1}'.format(outdir, data['fif_id']) + '_{0}.png'
    
    # Generate plots for the report
    print('Generating plots:')
    data['plt_flowchart'] = plot_flowchart(raw, savebase)
    data['plt_temporalsumsq'] = plot_channel_time_series(raw, savebase)
    data['plt_badchans'] = plot_sensors(raw, savebase)
    data['plt_channeldev'] = plot_channel_dists(raw, savebase)
    data['plt_spectra'], data['plt_zoomspectra'] = plot_spectra(raw, savebase)
    data['plt_digitisation'] = plot_digitisation_2d(raw, savebase)
    if level > 0:
        data['plt_artefacts_eog'] = plot_eog_summary(raw, savebase)
        data['plt_artefacts_ecg'] = plot_ecg_summary(raw, savebase)
        data['plt_artefacts_ica'] = plot_bad_ica(raw, ica, savebase) if ica is not None else None
    if level > 1:
        filenames = plot_artefact_scan(raw, savebase)
        data.update(filenames)

    return data


def gen_report(infiles, outdir, preproc_config=None, level=1):
    """Generate web-report for a set of MNE data objects.

    Parameters
    ----------
    infiles : list of str
        List of paths to fif files.
    outdir : str
        Directory to save HTML report and figures to.
    preproc_config : dict
        Preprocessing to apply before generating the report.
    level : int
        0 - basic report.
        1 - basic report with EOG, ECG and ICA.
        2 - basic report with EOG, ECG, ICA and an artefact scan.
    """

    # Validate input files and directory to save html file and plots to
    infiles, outnames, good_files = process_file_inputs(infiles)
    outdir = validate_outdir(outdir)

    # Load config for preprocessing to apply
    if preproc_config is not None:
        config = load_config(preproc_config)

    # Hyperlink to each panel on the page
    s = "<a href='#{0}'>{1}</a><br />"
    toplinks = [s.format(outnames[ii], infiles[ii]) for ii in range(len(infiles))]
    toplinks = '\n'.join(toplinks)

    # Load HTML template
    panels = []
    panel_template = load_template('panel')

    # Generate a panel for each file
    for infile in infiles:

        # Load preprocessed data file
        raw = import_data(infile)

        # Load ICA file if it exists
        # TODO: could potentially be incorporated in 'import_data'
        if os.path.exists(infile.replace('preproc_raw.fif', 'ica.fif')):
            ica = mne.preprocessing.read_ica(infile.replace('preproc_raw.fif', 'ica.fif'))
        else:
            ica = None

        # Apply some preprocessing if a config has been passed
        if preproc_config is not None:
            dataset = run_proc_chain(raw, config)
            raw = dataset['raw']
            ica = dataset['ica']

        # Generate data and plots for the panel
        data = gen_html_data(raw, ica, outdir, level)

        # Render the panel
        panels.append(panel_template.render(data=data))

    # Render the full page
    page_template = load_template('raw_report')
    page = page_template.render(panels=panels, toplinks=toplinks, level=level)

    # Write the output file
    outpath = '{0}/osl_raw_report.html'.format(outdir)
    with open(outpath, 'w') as f:
        f.write(page)
    print('REPORT :', outpath)


def load_template(tname):
    basedir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(basedir, 'templates', '{0}.html'.format(tname))
    with open(fname, 'r') as file:
        template = Template(file.read())
    return template


# ----------------------------------------------------------------------------------
# Scan stats and figures

def plot_flowchart(raw, savebase=None):
    """Plots preprocessing flowchart(s)"""
    
    # Get config info from raw.info['description']
    config_list = get_config_from_fif(raw)

    # Number of subplots, i.e. the number of times osl preprocessing was applied
    nrows = len(config_list)

    if nrows == 0:
        # No preprocessing was applied
        return None

    # Plot flowchart in subplots
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 6*(nrows+1)))

    cnt=0
    for config in config_list:
        if type(ax)==np.ndarray:
            axes=ax[cnt]
            title=f"OSL Preprocessing Stage {cnt+1}"
        else:
            axes=ax
            title=None
        fig, axes = plot_preproc_flowchart(config, fig=fig, ax=axes, title=title)
        cnt=cnt+1

    # Save figure
    figname = savebase.format('flowchart')
    print(figname)
    fig.savefig(figname, dpi=150, transparent=True)
    plt.close(fig)

    # Return the filename
    filebase = os.path.split(savebase)[1]
    return filebase.format('flowchart')

def plot_channel_time_series(raw, savebase=None):
    """Plots sum-square time courses."""

    # Raw data
    channel_types = {
        'Magnetometers': mne.pick_types(raw.info, meg='mag'),
        'Gradiometers': mne.pick_types(raw.info, meg='grad'),
        'EEG': mne.pick_types(raw.info, meg=False, eeg=True),
    }
    t = raw.times
    x = raw.get_data()

    # Number of subplots, i.e. the number of different channel types in the fif file
    nrows = 0
    for _, c in channel_types.items():
        if len(c) > 0:
            nrows += 1

    if nrows == 0:
        return 'No MEG or EEG channels.'

    # Make sum-square plots
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(16, 4))
    row = 0
    for name, chan_inds in channel_types.items():
        if len(chan_inds) == 0:
            continue
        ss = np.sum(x[chan_inds] ** 2, axis=0)
        ss = uniform_filter1d(ss, int(raw.info['sfreq']))
        ax[row].plot(t, ss)
        ax[row].legend([name], frameon=False)
        ax[row].set_xlim(t[0], t[-1])
        row += 1
    ax[0].set_title('Sum-Square Across Channels')
    ax[-1].set_xlabel('Time (seconds)')

    # Save
    if savebase is not None:
        plt.tight_layout()
        figname = savebase.format('temporal_sumsq')
        print(figname)
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

    # Return the filename
    filebase = os.path.split(savebase)[1]
    return filebase.format('temporal_sumsq')


def plot_sensors(raw, savebase=None):
    """Plots sensors with bad channels highlighted."""

    fig = raw.plot_sensors(show=False)
    figname = savebase.format('bad_chans')
    print(figname)
    fig.savefig(figname, dpi=150, transparent=True)
    plt.close(fig)

    # Return the filename
    filebase = os.path.split(savebase)[1]
    return filebase.format('bad_chans')


def plot_channel_dists(raw, savebase=None):
    """Plot distributions of temporal standard deviation."""

    # Raw data
    channel_types = {
        'Magnetometers': mne.pick_types(raw.info, meg='mag'),
        'Gradiometers': mne.pick_types(raw.info, meg='grad'),
        'EEG': mne.pick_types(raw.info, meg=False, eeg=True),
    }
    t = raw.times
    x = raw.get_data()

    # Number of subplots, i.e. the number of different channel types in the fif file
    ncols = 0
    for _, c in channel_types.items():
        if len(c) > 0:
            ncols += 1

    if ncols == 0:
        return 'No MEG or EEG channels.'
    
    # Make plots
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(12, 4))
    row = 0
    for name, chan_inds in channel_types.items():
        if len(chan_inds) == 0:
            continue
        ax[row].hist(x[chan_inds, :].std(axis=1), bins=24, histtype='step')
        ax[row].legend(['Temporal Std-Dev'], frameon=False)
        ax[row].set_xlabel('Std-Dev')
        ax[row].set_ylabel('Channel Count')
        ax[row].set_title(name)
        row += 1

    # Save
    if savebase is not None:
        plt.tight_layout()
        figname = savebase.format('channel_dev')
        print(figname)
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

    # Return the filename
    filebase = os.path.split(savebase)[1]
    return filebase.format('channel_dev')


def plot_spectra(raw, savebase=None):
    """Plot power spectra for each sensor modality."""

    # Plot spectra
    fig = raw.plot_psd(show=False, verbose=0)
    fig.set_size_inches(8, 7)

    # Save full spectra
    if savebase is not None:
        figname = savebase.format('spectra_full')
        print(figname)
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

    # Plot zoomed in spectra
    fig = raw.plot_psd(show=False, fmin=1, fmax=48, verbose=0)
    fig.set_size_inches(8, 7)

    # Save zoomed in spectra
    if savebase is not None:
        figname = savebase.format('spectra_zoom')
        print(figname)
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

    # Return filenames
    filebase = os.path.split(savebase)[1]
    return filebase.format('spectra_full'), filebase.format('spectra_zoom')


def plot_digitisation_2d(raw, savebase=None):
    """Plots the digitisation and headshape."""

    # Make plot
    fig = plt.figure(figsize=(16, 4))

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
    legend_elements = [Line2D([0], [0], marker='^', color='w', lw=4, label='Fiducial', markerfacecolor='r',markersize=14),
                       Line2D([0], [0], marker='o', color='w', label='HPI', markerfacecolor='m', markersize=14),
                       Line2D([0], [0], marker='*', color='w', label='EEG', markerfacecolor='g', markersize=14),
                       Line2D([0], [0], marker='.', color='w', label='Headshape', markerfacecolor='b', markersize=14)]
    plt.legend(handles=legend_elements, loc='center', frameon=False)

    if savebase is not None:
        plt.tight_layout()
        figname = savebase.format('digitisation')
        print(figname)
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

    # Return the filename
    filebase = os.path.split(savebase)[1]
    return filebase.format('digitisation')


def plot_headmovement(raw, savebase=None):
    """Plot headmovement - WORK IN PROGRESS... seems v-slow atm"""
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=False)
    fig = mne.viz.plot_head_positions(head_pos, mode='traces')
    if savebase is not None:
        figname = savebase.format('headpos')
        print(figname)
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)


def plot_eog_summary(raw, savebase=None):
    """Plot raw EOG time series."""

    # Get the raw EOG data
    chan_inds = mne.pick_types(raw.info, eog=True)
    if len(chan_inds) == 0:
        return 'EOG Channel Not Found'
    t = raw.times
    x = raw.get_data(chan_inds).T

    # Make the plot
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 5))
    ax[0].set_title('Full Time Series')
    ax[0].plot(t, x)
    ax[0].set_xlim([t[0], t[-1]])

    # Plot the first 30 seconds on the second row
    n = int(raw.info['sfreq'] * 30)
    ax[1].set_title('First 30 Seconds')
    ax[1].plot(t[:n], x[:n])
    ax[1].set_xlabel('Time (s)')
    ax[1].set_xlim([t[0], t[n]])

    # Save
    if savebase is not None:
        plt.tight_layout()
        figname = savebase.format('EOG')
        print(figname)
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

    # Return the filename
    filebase = os.path.split(savebase)[1]
    return filebase.format('EOG')


def plot_ecg_summary(raw, savebase=None):
    """Plot ECG summary."""

    # Get ECG data
    chan_inds = mne.pick_types(raw.info, ecg=True)
    if len(chan_inds) == 0:
        return 'ECG Channel Not Found'
    x = raw.get_data(chan_inds)
    if np.abs(x.min()) > x.max():
        x = -x

    # Process first ECG channel
    signals, info = nk.ecg_process(x[0, :], sampling_rate=raw.info['sfreq'])
    nk.ecg_plot(signals, sampling_rate=raw.info['sfreq'])
    fig = plt.gcf()
    fig.set_size_inches(16, 7)

    # Save
    if savebase is not None:
        plt.tight_layout()
        figname = savebase.format('ECG')
        print(figname)
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

    # Return the filename
    filebase = os.path.split(savebase)[1]
    return filebase.format('ECG')


def plot_bad_ica(raw, ica, savebase):
    """Plot ICA characteristics for rejected components."""

    exclude_uniq = np.sort(np.unique(ica.exclude))
    nbad = len(exclude_uniq)

    # Create figure
    fig = plt.figure(figsize=(16, 5 * nbad), facecolor=[0.95] * 3)
    axes = []
    for i in np.arange(nbad):
        lowerlimit = 0.1 + i / (nbad * 1.1)
        multiplier = nbad * 1.3

        # Create axis for subplot
        # adapted from mne/viz/ica._create_properties_layout
        axes_params = (('topomap', [0.08, lowerlimit + 0.5 / multiplier, 0.3, 0.45 / multiplier]),
                       ('image', [0.5, lowerlimit + 0.6 / multiplier, 0.45, 0.35 / multiplier]),
                       ('erp', [0.5, lowerlimit + 0.5 / multiplier, 0.45, 0.1 / multiplier]),
                       ('spectrum', [0.08, lowerlimit + 0.1 / multiplier, 0.32, 0.3 / multiplier]),
                       ('variance', [0.5, lowerlimit + 0.025 / multiplier, 0.45, 0.25 / multiplier]))
        axes += [[fig.add_axes(loc, label=name) for name, loc in axes_params]]

        ica.plot_properties(raw, picks=exclude_uniq[i], axes=axes[i], show=False, verbose=0)

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
        figname = savebase.format('ica')
        print(figname)
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

    # Return the filename
    filebase = os.path.split(savebase)[1]
    return filebase.format('ica')


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

    # Return filenames
    filebase = os.path.split(savebase)[1]
    filenames = {
        'plt_eyemove_grad': filebase.format('eyemove_grad'),
        'plt_eyemove_mag': filebase.format('eyemove_mag'),
        'plt_eyemove_eog': filebase.format('eyemove_eog'),
        'plt_blink_grad': filebase.format('blink_grad'),
        'plt_blink_mag': filebase.format('blink_mag'),
        'plt_blink_eog': filebase.format('blink_eog'),
        'plt_swallow_grad': filebase.format('swallow_grad'),
        'plt_swallow_mag': filebase.format('swallow_mag'),
        'plt_breathe_grad': filebase.format('breathe_grad'),
        'plt_breathe_mag': filebase.format('breathe_mag'),
        'plt_shrug_grad': filebase.format('shrug_grad'),
        'plt_shrug_mag': filebase.format('shrug_mag'),
        'plt_clench_grad': filebase.format('clench_grad'),
        'plt_clench_mag': filebase.format('clench_mag'),
        'plt_buttonpress_grad': filebase.format('buttonpress_grad'),
        'plt_buttonpress_mag': filebase.format('buttonpress_mag'),
    }
    return filenames


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
    try:
        nhpi = len(raw.info['hpi_meas'][0]['hpi_coils'])
        print('{0} HPI coils acquired'.format(nhpi))
    except:
        print("No HPI info in fif file")

    # Number of channels acquired
    nchans = raw.info['nchan']
    print('{0} channels acquired'.format(nchans))

    # Breakdown channels into channel types
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

    # Bad segments
    durs = np.array([r['duration'] for r in raw.annotations])
    full_dur = raw.n_times/raw.info['sfreq']
    types = [r['description'] for r in raw.annotations]

    for modality in ['grad', 'mag', 'eeg']:
        inds = [s.find(modality) > 0 for s in types]
        mod_dur = np.sum(durs[inds])
        pc = (mod_dur / full_dur) * 100
        s = 'Modality {0} - {1:02f}/{2} seconds rejected     ({3:02f}%)'
        print(s.format(modality, mod_dur, full_dur, pc))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Run a quality control summary on data.')
    parser.add_argument('files', type=str, nargs='+',
                        help='plain text file containing full paths to files to be processed')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Path to output directory to save data in')
    parser.add_argument('--preproc_config', type=str,
                        help='yaml defining preprocessing')
    parser.add_argument('--level', type=int, default=1, help='Level of reporting')

    args = parser.parse_args(argv)

    gen_report(args.files, args.outdir, args.preproc_config, args.level)


if __name__ == '__main__':
    main()
