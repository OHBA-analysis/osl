#!/usr/bin/env python3

# vim: set expandtab ts=4 sw=4:

import mne
import sys
import sails
import numpy as np
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Global (within script...) variables containing html templates. Should
# probably load these from a file or something more organised.

# Overall page template
html_base = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>NTAD-QC</title>
    <style>
        body { background-color: White; }
        p { color: #fff; }
        div.figbox { width: 50%; display: table-cell; text-align: center; }
        div.tablebox { width: 30%; display: table-cell; text-align: left; margin; 20px }
        th, td { border-bottom: 1px solid #ddd; }
        table { border-collapse: collapse; width: 75%; }
    </style>
</head>
<body>


<div style='width:80%; margin: 100px; text-align: center' >
    <h2>NTAD Quality Check Report</h2>
    @headershere
</div>

@scanshere

</body>
"""

# Single file report template - added per-file at @scanshere in html_base
fif_report = """
<div style='width: 100%; margin-left: 100px; margin-top: 100px'>
    <h3 id="@fif_id">@filename</h3>
    <b>Datafile:</b> @filename</br>
    <b>Project name:</b> @projname</br>
    <b>Experimenter:</b> @experimenter</br>
    <b>Data acquired:</b> @meas_date</br>
    @acq_samples samples at @acq_sfreq Hz - @acq_duration seconds
</div>

<div style="width: 70%; display: table; margin-left: 100px; margin-top: 50px">
    <div style="display: table-row; height: 100px;">
        <div class='tablebox'>
            @chantable
    </div>
        <div class='tablebox'>
            @digitable
    </div>
        <div class='tablebox'>
            @eventtable
        </div>
    </div>
</div>

<div style="width: 80%; padding-top: 50px">
    <img src="@plt_temporaldev" alt="" style='max-width: 1536px'/>
</div>

<div style="width: 80%; padding-top: 50px">
    <img src="@plt_channeldev" alt="" style='max-width: 1536px'/>
</div>

<div style="width: 80%; display: table; padding-top: 50px">
    <div style="display: table-row; height: 100px;">
        <div class='figbox'>
            <h3>Full Power Spectra</h3>
            <img src="@plt_spectra" alt="" style='max-width: 768px'/>
        </div>
        <div class='figbox'>
            <h3>1-48Hz Power Spectra</h3>
            <img src="@plt_zoom_spectra" alt="", style='max-width: 768px'/>
        </div>
    </div>
</div>

<div style="width: 80%; display: table;">
    <div style="display: table-row; height: 100px;">
        <div class='figbox'>
            <h3>Artefact Channels</h3>
            <img src="@plt_artefacts" alt="" style='max-width: 768px'/>
        </div>
        <div class='figbox'>
            <h3>First 30 Seconds of Artefact Channels</h3>
            <img src="@plt_zoom_artefacts" alt="" style='max-width: 768px'/>
        </div>
    </div>
</div>
"""


# ----------------------------------------------------------------------------------
# Report generation

def get_header_id(raw):
    """Extract scan name from MNE data object."""
    return raw.filenames[0].split('/')[-1].strip('.fif')


def gen_fif_html(raw, outf=None, fif_id=None, gen_plots=True):
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
    digs = [d['kind'] for d in raw.info['dig']]
    d, dcounts = np.unique(digs, return_counts=True)
    x['digitable'] = tabulate(np.c_[dig_codes, dcounts], tablefmt='html',
                              headers=['Digitisation Stage', 'Points Acquired'])

    ev = mne.find_events(raw, min_duration=5/raw.info['sfreq'], verbose=False)
    ev, evcounts = np.unique(ev[:, 2], return_counts=True)
    x['eventtable'] = tabulate(np.c_[ev, evcounts], tablefmt='html',
                               headers=['Event Code', 'Value'])

    savebase = '{0}/{1}'.format(outf, x['fif_id']) + '_{0}.png'
    if gen_plots:
        plot_artefact_channels(raw, savebase=savebase)
        plot_spectra(raw, savebase=savebase)
        plot_channel_dists(raw, savebase=savebase)
    plt.close('all')

    x['plt_channeldev'] = savebase.format('channel_dev')
    x['plt_temporaldev'] = savebase.format('temporal_dev')
    x['plt_artefacts'] = savebase.format('artefacts')
    x['plt_zoom_artefacts'] = savebase.format('artefacts_zoom')
    x['plt_spectra'] = savebase.format('spectra_full')
    x['plt_zoom_spectra'] = savebase.format('spectra_zoom')

    # Replace the target string
    filedata = fif_report
    for key in x.keys():
        filedata = filedata.replace("@{0}".format(key), str(x[key]))

    return filedata


def gen_report(raws, outdir):
    """Generate web-report for a set of MNE data objects."""
    html = [gen_fif_html(raw, outf=outdir) for raw in raws]
    html = '\n'.join(html)

    names = [raw.filenames[0] for raw in raws]
    fif_ids = [get_header_id(raw) for raw in raws]

    s = "<a href='#{0}'>{1}</a><br />"
    top_links = [s.format(fif_ids[ii], names[ii]) for ii in range(len(names))]
    top_links = '\n'.join(top_links)

    # Replace the target string
    global html_base
    html_base = html_base.replace("@scanshere", html)
    html_base = html_base.replace("@headershere", top_links)

    # Write the file out again
    outpath = '{0}/ntad_qc_index.html'.format(outdir)
    with open(outpath, 'w') as f:
        f.write(html_base)
    print(outpath)


# ----------------------------------------------------------------------------------
# Scan processing


def base_sensor_proc(raw):
    """Run a very simple preprocessing."""
    raw.resample(400)
    raw.filter(0.5, 125, picks=['meg', 'eeg'])
    gra = get_badseg_annotations(raw, picks='grad')
    mag = get_badseg_annotations(raw, picks='mag')
    eeg = get_badseg_annotations(raw, picks='eeg')
    raw.set_annotations(gra+mag+eeg)

    return raw


def get_badseg_annotations(raw, segment_len=400, picks=None):
    """Set bad segments in MNE object."""
    bdinds = sails.utils.detect_artefacts(raw.get_data(picks=picks), 1,
                                          reject_mode='segments',
                                          segment_len=segment_len,
                                          ret_mode='bad_inds')

    onsets = np.where(np.diff(bdinds.astype(float)) == 1)[0]
    if bdinds[0] is True:
        onsets = np.r_[0, onsets]
    offsets = np.where(np.diff(bdinds.astype(float)) == -1)[0]

    if bdinds[-1] is True:
        offsets = np.r_[offsets, len(bdinds)-1]
    assert(len(onsets) == len(offsets))
    durations = offsets - onsets
    descriptions = np.repeat('bad_segment_{0}'.format(picks), len(onsets))

    onsets = onsets / raw.info['sfreq']
    durations = durations / raw.info['sfreq']

    return mne.Annotations(onsets, durations, descriptions)


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
    fig.set_size_inches(10, 10)
    if savebase is not None:
        fig.savefig(savebase.format('spectra_full'), dpi=150, transparent=True)

    fig = raw.plot_psd(show=False, fmin=1, fmax=48)
    fig.set_size_inches(10, 10)
    if savebase is not None:
        fig.savefig(savebase.format('spectra_zoom'), dpi=150, transparent=True)


def plot_channel_dists(raw, savebase=None):
    """Plot summary distributions of sensors."""
    fig = plt.figure(figsize=(16, 4))
    plt.subplot(131)
    inds = mne.pick_types(raw.info, meg='mag')
    plt.hist(raw.get_data()[inds, :].std(axis=1))
    plt.xlabel('St-Dev')
    plt.ylabel('Channel Count')
    plt.title('Magnetometers temporal std-dev')
    plt.subplot(132)
    inds = mne.pick_types(raw.info, meg='grad')
    plt.hist(raw.get_data()[inds, :].std(axis=1))
    plt.title('Gradiometers temporal st-dev')
    plt.xlabel('St-Dev')
    plt.subplot(133)
    inds = mne.pick_types(raw.info, meg=False, eeg=True)
    plt.hist(raw.get_data()[inds, :].std(axis=1))
    plt.title('EEG temporal st-dev')
    plt.xlabel('St-Dev')
    if savebase is not None:
        fig.savefig(savebase.format('channel_dev'), dpi=150, transparent=True)

    fig = plt.figure(figsize=(16, 4))
    plt.subplot(131)
    inds = mne.pick_types(raw.info, meg='mag')
    plt.plot(raw.times, raw.get_data()[inds, :].std(axis=0))
    plt.title('EEG channel st-dev')
    plt.ylabel('St-Dev over Channels')
    plt.title('Magnetometers channel std-dev')
    plt.xlabel('Time (seconds)')
    plt.subplot(132)
    inds = mne.pick_types(raw.info, meg='grad')
    plt.plot(raw.times, raw.get_data()[inds, :].std(axis=0))
    plt.title('EEG channel st-dev')
    plt.title('Gradiometers channel st-dev')
    plt.xlabel('Time (seconds)')
    plt.subplot(133)
    inds = mne.pick_types(raw.info, meg=False, eeg=True)
    plt.plot(raw.times, raw.get_data()[inds, :].std(axis=0))
    plt.title('EEG channel st-dev')
    plt.xlabel('Time (seconds)')
    if savebase is not None:
        fig.savefig(savebase.format('temporal_dev'), dpi=150, transparent=True)


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
    parser.add_argument('files', type=str,
                        help='plain text file containing full paths to files to be processed')
    parser.add_argument('outdir', type=str,
                        help='Path to output directory to save data in')

    args = parser.parse_args(argv)
    print(args)

    # -------------------------------------------

    with open(args.files, 'r') as f:
        infifs = f.readlines()
    infifs = [fif.strip('\n') for fif in infifs]

    raws = [mne.io.read_raw_fif(m) for m in infifs]

    gen_report(raws, args.outdir)


# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
