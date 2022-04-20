import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import ConnectionPatch
from scipy import signal

#%% ----------------------------------------------
#
# Main plotting funcs


def plot_sensor_data(data, raw, ax=None, xvect=None, lw=0.5,
                     xticks=None, xticklabels=None,
                     sensor_cols=True, base=1):
    """Plot sensorspace time-series data in an axis."""
    if xvect is None:
        xvect = np.arange(obs.shape[0])
    if ax is None:
        ax = plt.subplot(111)

    fx, xticklabels, xticks = prep_scaled_freq(base, xvect)

    if sensor_cols:
        colors, pos, outlines = get_mne_sensor_cols(raw)
    else:
        colors = None

    plot_with_cols(ax, data, fx, colors, lw=lw)
    ax.set_xlim(fx[0], fx[-1])

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)


def plot_sensor_timeseries(timeseries, raw, xvect=None, ax=None, sensor_proj=False,
                          xticks=None, xticklabels=None, lw=0.5,
                          sensor_cols=True, ylabel=None):
    """Plot sensorspace power spectrum data in an axis."""
    if xvect is None:
        xvect = np.arange(timeseries.shape[0])
    if ax is None:
        ax = plt.subplot(111)

    plot_sensor_data(timeseries, raw, ax=ax, sensor_cols=sensor_cols, lw=lw,
                     xvect=xvect, xticks=xticks, xticklabels=xticklabels)
    decorate_timseries(ax, ylabel=ylabel)
    ax.set_ylim(timeseries.min()*1.1, timeseries.max()*1.1)

    if sensor_proj:
        axins = ax.inset_axes([0.6, 0.6, 0.37, 0.37])
        plot_channel_layout(axins, raw)


def plot_sensor_spectrum(psd, raw, xvect=None, ax=None, sensor_proj=False,
                         xticks=None, xticklabels=None, lw=0.5,
                         sensor_cols=True, base=1, ylabel=None):
    """Plot sensorspace power spectrum data in an axis."""
    if xvect is None:
        xvect = np.arange(obs.shape[0])
    if ax is None:
        ax = plt.subplot(111)

    plot_sensor_data(psd, raw, ax=ax, base=base, sensor_cols=sensor_cols, lw=lw,
                     xvect=xvect, xticks=xticks, xticklabels=xticklabels)
    decorate_spectrum(ax, ylabel=ylabel)
    ax.set_ylim(psd.min())

    if sensor_proj:
        axins = ax.inset_axes([0.6, 0.6, 0.37, 0.37])
        plot_channel_layout(axins, raw)


def plot_channel_layout(ax, raw, size=30, marker='o'):
    """Plot a schematic 2d sensor layout."""
    ax.set_adjustable('box')
    ax.set_aspect('equal')

    colors, pos, outlines = get_mne_sensor_cols(raw)
    pos_x, pos_y = pos.T
    mne.viz.evoked._prepare_topomap(pos, ax, check_nonzero=False)
    ax.scatter(pos_x, pos_y,
               color=colors, s=size * .8,
               marker=marker, zorder=1)
    mne.viz.evoked._draw_outlines(ax, outlines)


def plot_joint_spectrum(ax, xvect, psd, raw, freqs='auto', base=1,
                        topo_scale='joint', lw=0.5, ylabel='Power', title=''):
    """Plot sensorspace power spectra with illustrative topographies."""
    plot_sensor_spectrum(ax, psd, raw, xvect, base=base, lw=lw, ylabel=ylabel)
    fx, xtl, xt = prep_scaled_freq(base, xvect)

    if freqs == 'auto':
        freqs = signal.find_peaks(psd.mean(axis=1), distance=10)[0]
        if 0 not in freqs:
            freqs = np.r_[0, freqs]
    else:
        # Convert Hz to samples in freq dim
        freqs = [np.argmin(np.abs(xvect - f)) for f in freqs]

    topo_centres = np.linspace(0, 1, len(freqs)+2)[1:-1]
    topo_width = 0.4

    shade = [0.7, 0.7, 0.7]

    if topo_scale is 'joint':
        vmin = psd.min()
        vmax = psd.max()
    else:
        vmin = None
        vmax = None

    for idx in range(len(freqs)):
        # Create topomap axis
        topo_pos = [topo_centres[idx] - 0.2, 1.2, 0.4, 0.4]
        topo = ax.inset_axes(topo_pos)

        # Draw topomap itself
        mne.viz.plot_topomap(psd[freqs[idx], :], raw.info, axes=topo,
                             cmap='RdBu_r', vmin=vmin, vmax=vmax)

        # Add angled connecting line
        xy = (fx[freqs[idx]], ax.get_ylim()[1])
        con = ConnectionPatch(xyA=xy, xyB=(0, topo.get_ylim()[0]),
                              coordsA=ax.transData, coordsB=topo.transData,
                              axesA=ax, axesB=topo, color=shade, lw=2)
        ax.get_figure().add_artist(con)

    # Add vertical lines
    ax.vlines(fx[freqs], ax.get_ylim()[0], ax.get_ylim()[1], color=shade, lw=2)

    ax.set_title(title)


def plot_sensorspace_timeseries_clusters(xvect, dat, raw, P, ax=None, ylabel='Power',
                                         topo_scale='joint', base=1, lw=0.5, title=None, thresh=95):
    """Plot the result of non-parametric cluster permutations on a sensorspace time-series."""
    if ax is None:
        ax = plt.subplot(212)

    from matplotlib.patches import ConnectionPatch
    clu, obs = P.get_sig_clusters(thresh, dat)
    print(obs.shape)
    if xvect is None:
        xvect = np.arange(obs.shape[0])

    plot_sensor_timeseries(obs, raw, xvect=xvect, ax=ax, lw=lw, ylabel=ylabel)
    fx, xtl, xt = prep_scaled_freq(1, xvect)

    shade = [0.7, 0.7, 0.7]
    xf = -0.03

    # sort clusters by ascending freq
    forder = np.argsort([c[2][0].mean() for c in clu])
    clu = [clu[c] for c in forder]

    if topo_scale is 'joint':
        vmin = obs.min()
        vmax = obs.max()
    else:
        vmin = None
        vmax = None

    topo_centres = np.linspace(0, 1, len(clu)+2)[1:-1]
    topo_width = 0.4

    if len(clu) == 0:
        # put up an empty axes anyway
        topo_pos = [0.3, 1.2, 0.4, 0.4]
        topo = ax.inset_axes(topo_pos, frame_on=False)
        topo.set_xticks([])
        topo.set_yticks([])

    for c in range(len(clu)):
        inds = np.where(clu==c+1)[0]
        channels = np.zeros((obs.shape[1], ))
        channels[clu[c][2][1]] = 1
        if len(channels) == 204:
            channels = np.logical_and(channels[::2], channels[1::2])
        times = np.zeros((obs.shape[0], ))
        times[clu[c][2][0]] = 1
        tinds = np.where(times)[0]
        if len(tinds) == 1:
            tinds = [tinds[0], tinds[0]+1]
        ax.axvspan(fx[tinds[0]], fx[tinds[-1]], facecolor=shade, alpha=0.5)

        topo_pos = [topo_centres[c] - 0.2, 1.2, 0.4, 0.4]
        topo = ax.inset_axes(topo_pos)
        mne.viz.plot_topomap(obs[tinds, :].mean(axis=0), raw.info, axes=topo,
                             cmap='RdBu_r', vmin=vmin, vmax=vmax, mask=channels.astype(int))

        xy = (fx[tinds].mean(), ax.get_ylim()[1])
        con = ConnectionPatch(xyA=xy, xyB=(0, topo.get_ylim()[0]),
                              coordsA=ax.transData, coordsB=topo.transData,
                              axesA=ax, axesB=topo, color=shade)
        plt.gcf().add_artist(con)

    ax.set_title(title)


def plot_sensorspace_spectrum_clusters(xvect, dat, raw, P, ax=None, ylabel='Power',
                                       topo_scale='joint', base=1, lw=0.5, title=None, thresh=95):
    """Plot the result of non-parametric cluster permutations on a sensorspace power spectrum."""
    if ax is None:
        ax = plt.subplot(111)

    from matplotlib.patches import ConnectionPatch
    clu, obs = P.get_sig_clusters(thresh, dat)
    print(obs.shape)
    if xvect is None:
        xvect = np.arange(obs.shape[0])

    plot_sensor_spectrum(ax, obs, raw, xvect, base=base, lw=lw, ylabel=ylabel)
    fx, xtl, xt = prep_scaled_freq(base, xvect)

    shade = [0.7, 0.7, 0.7]
    xf = -0.03

    # sort clusters by ascending freq
    forder = np.argsort([c[2][0].mean() for c in clu])
    clu = [clu[c] for c in forder]

    if topo_scale is 'joint':
        vmin = obs.min()
        vmax = obs.max()
    else:
        vmin = None
        vmax = None

    topo_centres = np.linspace(0, 1, len(clu)+2)[1:-1]
    topo_width = 0.4

    if len(clu) == 0:
        # put up an empty axes anyway
        topo_pos = [0.3, 1.2, 0.4, 0.4]
        topo = ax.inset_axes(topo_pos, frame_on=False)
        topo.set_xticks([])
        topo.set_yticks([])

    for c in range(len(clu)):
        inds = np.where(clu==c+1)[0]
        channels = np.zeros((obs.shape[1], ))
        channels[clu[c][2][1]] = 1
        if len(channels) == 204:
            channels = np.logical_and(channels[::2], channels[1::2])
        times = np.zeros((obs.shape[0], ))
        times[clu[c][2][0]] = 1
        tinds = np.where(times)[0]
        if len(tinds) == 1:
            tinds = [tinds[0], tinds[0]+1]
        ax.axvspan(fx[tinds[0]], fx[tinds[-1]], facecolor=shade, alpha=0.5)

        topo_pos = [topo_centres[c] - 0.2, 1.2, 0.4, 0.4]
        topo = ax.inset_axes(topo_pos)
        mne.viz.plot_topomap(obs[tinds, :].mean(axis=0), raw.info, axes=topo,
                             cmap='RdBu_r', vmin=vmin, vmax=vmax, mask=channels.astype(int))

        xy = (fx[tinds].mean(), ax.get_ylim()[1])
        con = ConnectionPatch(xyA=xy, xyB=(0, topo.get_ylim()[0]),
                              coordsA=ax.transData, coordsB=topo.transData,
                              axesA=ax, axesB=topo, color=shade)
        plt.gcf().add_artist(con)

    ax.set_title(title)


#%% -----------------------------------------------------------
# Helpers

# This should probably be in a general plotting utils submodule at some point
def subpanel_label(ax, label, xf=-0.1, yf=1.1):
    """Add a simple A,B,C,D type subpanel label to an axis."""
    ypos = ax.get_ylim()[0]
    yyrange = np.diff(ax.get_ylim())[0]
    ypos = (yyrange * yf) + ypos
    # Compute letter position as proportion of full xrange.
    xpos = ax.get_xlim()[0]
    xxrange = np.diff(ax.get_xlim())[0]
    xpos = (xxrange * xf) + xpos
    ax.text(xpos, ypos, label, horizontalalignment='center',
            verticalalignment='center', fontsize=20, fontweight='bold')


def decorate_spectrum(ax, ylabel='Power'):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(ylabel)


def decorate_timseries(ax, ylabel='Amplitude'):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)


def plot_with_cols(ax, data, xvect, cols=None, lw=0.5):
    """Plot time-series with given colours."""
    if cols is not None:
        for ii in range(data.shape[1]):
            ax.plot(xvect, data[:, ii], lw=lw, color=cols[ii, :])
    else:
        ax.plot(xvect, data, lw=lw)


def prep_scaled_freq(base, freq_vect):
    """Assuming ephy freq ranges for now - around 1-40Hz"""
    fx = freq_vect**base
    if base <= 1:
        nticks = int(np.floor(np.sqrt(freq_vect[-1])))
        #ftick = np.array([2**ii for ii in range(6)])
        ftick = np.array([ii**2 for ii in range(1,nticks+1)])
        ftickscaled = ftick**base
    else:
        # Stick with automatic scales
        ftick = None
        ftickscaled = None
    return fx, ftick, ftickscaled


def get_mne_sensor_cols(raw, picks=None):
    """Get sensor location coded colours in MNE-style."""
    if picks is not None:
        raw.pick_types(**picks)

    chs = [raw.info['chs'][i] for i in range(len(raw.info['chs']))]
    locs3d = np.array([ch['loc'][:3] for ch in chs])
    x, y, z = locs3d.T
    colors = mne.viz.evoked._rgb(x, y, z)
    pos, outlines = mne.viz.evoked._get_pos_outlines(raw.info,
                                                     range(len(raw.info['chs'])),
                                                     sphere=None)

    return colors, pos, outlines
