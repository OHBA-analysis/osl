# Used as alternative to MNE-python's ica.plot_sources(inst), presenting an interactive figure
# with ICA time courses and topographies (of both sensor types if applicable). By clicking on
# a time course, it is marked as artifact.
# Use as:
# from osl_plot_ica import *
# plot_ica(ica,raw)
# 2021 - Mats van Es

#%% MVE: THIS COMES FROM MNE/VIZ/ICA/PLOT_ICA_SOURCES
def plot_ica(ica, inst, picks=None, start=None,
                     stop=None, title=None, show=True, block=False,
                     show_first_samp=False, show_scrollbars=True, n_channels=10,# MVE: EXTRA INPUT 'N_CHANNELS'
                     bad_labels_list=['eog', 'ecg', 'emg', 'hardware', 'other']): # label bad components
    """Plot estimated latent sources given the unmixing matriX & Project mixing matrix on interpolated sensor topography.

    INFO FROM PLOT_ICA_SOURCES
    Typical usecases:

    1. plot evolution of latent sources over time based on (Raw input)
    2. plot latent source around event related time windows (Epochs input)
    3. plot time-locking in ICA space (Evoked input)

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA solution.
    inst : instance of mne.io.Raw, mne.Epochs, mne.Evoked
        The object to plot the sources from.
    %(picks_base)s all sources in the order as fitted.
    start : int
        X-axis start index. If None, from the beginning.
    stop : int
        X-axis stop index. If None, next 10 are shown, in case of evoked to the
        end.
    title : str | None
        The window title. If None a default is provided.
    show : bool
        Show figure if True.
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for interactive selection of components in raw and epoch
        plotter. For evoked, this parameter has no effect. Defaults to False.
    show_first_samp : bool
        If True, show time axis relative to the ``raw.first_samp``.
    %(show_scrollbars)s
    n_channels: int
        Number of channels to plot (default=10 or number of picks)

    Returns
    -------
    fig : instance of Figure
        The figure.

    Notes
    -----
    For raw and epoch instances, it is possible to select components for
    exclusion by clicking on the line. The selected components are added to
    ``ica.exclude`` on close.

    .. versionadded:: 0.10.0
    """

    from mne.io.base import BaseRaw
    from mne.epochs import BaseEpochs
    from mne.io.pick import _picks_to_idx

    exclude = ica.exclude
    picks = _picks_to_idx(ica.n_components_, picks, 'all')

    #"""Plot the ICA components as a RawArray or EpochsArray."""
    if isinstance(inst, (BaseRaw, BaseEpochs)):
        fig = _plot_sources(ica, inst, picks, exclude, start=start, stop=stop,
                            show=show, title=title, block=block,
                            show_first_samp=show_first_samp,
                            show_scrollbars=show_scrollbars, n_channels=n_channels, # MVE: give extra input n_channels
                            bad_labels_list=bad_labels_list) # label bad components
    # MVE: Evoked not yet working.
    # TODO: work on Evoked
    # elif isinstance(inst, Evoked):
    #     if start is not None or stop is not None:
    #         inst = inst.copy().crop(start, stop)
    #     sources = ica.get_sources(inst)
    #     fig = _plot_ica_sources_evoked(
    #         evoked=sources, picks=picks, exclude=exclude, title=title,
    #         labels=getattr(ica, 'labels_', None), show=show, ica=ica)
    else:
        raise ValueError('Data input must be of Raw or Epochs type')
    return fig

#%% MVE: THIS COMES FROM MNE/VIZ/ICA._PLOT_SOURCES
def _plot_sources(ica, inst, picks, exclude, start, stop, show, title, block,
                  show_scrollbars, show_first_samp, n_channels=10, bad_labels_list=['eog', 'ecg', 'emg', 'hardware', 'other']): # MVE: EXTRA INPUT N_CHANNELS
    from mne import EpochsArray, BaseEpochs
    from mne.io import RawArray, BaseRaw
    from mne.viz.utils import _compute_scalings, _make_event_color_dict
    from mne.io.meas_info import create_info

    # handle defaults / check arg validity
    is_raw = isinstance(inst, BaseRaw)
    is_epo = isinstance(inst, BaseEpochs)
    sfreq = inst.info['sfreq']
    color = _handle_default('color', (0., 0., 0.))
    units = _handle_default('units', None)
    scalings = (_compute_scalings(None, inst) if is_raw else
                _handle_default('scalings_plot_raw'))
    scalings['misc'] = 5.
    scalings['whitened'] = 1.
    unit_scalings = _handle_default('scalings', None)

    # data
    if is_raw:
        data = ica._transform_raw(inst, 0, len(inst.times))[picks]
    else:
        data = ica._transform_epochs(inst, concatenate=True)[picks]

    # events
    if is_epo:
        event_id_rev = {v: k for k, v in inst.event_id.items()}
        event_nums = inst.events[:, 2]
        event_color_dict = _make_event_color_dict(None, inst.events,
                                                  inst.event_id)

    # channel properties / trace order / picks
    ch_names = list(ica._ica_names)  # copy
    ch_types = ['misc' for _ in picks]

    # add EOG/ECG channels if present
    eog_chs = pick_types(inst.info, meg=False, eog=True, ref_meg=False)
    ecg_chs = pick_types(inst.info, meg=False, ecg=True, ref_meg=False)
    for eog_idx in eog_chs:
        ch_names.append(inst.ch_names[eog_idx])
        ch_types.append('eog')
    for ecg_idx in ecg_chs:
        ch_names.append(inst.ch_names[ecg_idx])
        ch_types.append('ecg')
    extra_picks = np.concatenate((eog_chs, ecg_chs)).astype(int)
    if len(extra_picks):
        if is_raw:
            eog_ecg_data, _ = inst[extra_picks, :]
        else:
            eog_ecg_data = np.concatenate(inst.get_data(extra_picks), axis=1)
        data = np.append(data, eog_ecg_data, axis=0)
    picks = np.concatenate(
        (picks, ica.n_components_ + np.arange(len(extra_picks))))
    ch_order = np.arange(len(picks))
    n_channels = min([n_channels, len(picks)])

    # create info
    info = create_info([ch_names[x] for x in picks], sfreq, ch_types=ch_types)
    info['meas_date'] = inst.info['meas_date']
    info['bads'] = [ch_names[x] for x in exclude]
    if is_raw:
        inst_array = RawArray(data, info, inst.first_samp)
        inst_array.set_annotations(inst.annotations)
    else:
        data = data.reshape(-1, len(inst), len(inst.times)).swapaxes(0, 1)
        inst_array = EpochsArray(data, info)

    # handle time dimension
    start = 0 if start is None else start
    _last = inst.times[-1] if is_raw else len(inst.events)
    stop = min(start + 10, _last) if stop is None else stop
    first_time = inst._first_time if show_first_samp else 0
    if is_raw:
        duration = stop - start
        start += first_time
    else:
        n_epochs = stop - start
        total_epochs = len(inst)
        epoch_n_times = len(inst.times)
        n_epochs = min(n_epochs, total_epochs)
        n_times = total_epochs * epoch_n_times
        duration = n_epochs * epoch_n_times / sfreq
        event_times = (np.arange(total_epochs) * epoch_n_times
                       + inst.time_as_index(0)) / sfreq
        # NB: this includes start and end of data:
        boundary_times = np.arange(total_epochs + 1) * epoch_n_times / sfreq
    if duration <= 0:
        raise RuntimeError('Stop must be larger than start.')

    # misc
    bad_color = (0.8, 0.8, 0.8)
    title = 'ICA components' if title is None else title

    params = dict(inst=inst_array,
                  ica=ica,
                  ica_inst=inst,
                  info=info,
                  # channels and channel order
                  ch_names=np.array(ch_names),
                  ch_types=np.array(ch_types),
                  ch_order=ch_order,
                  picks=picks,
                  n_channels=n_channels,
                  picks_data=list(),
                  bad_labels_list=bad_labels_list,
                  # time
                  t_start=start if is_raw else boundary_times[start],
                  duration=duration,
                  n_times=inst.n_times if is_raw else n_times,
                  first_time=first_time,
                  decim=1,
                  # events
                  event_times=None if is_raw else event_times,
                  # preprocessing
                  projs=list(),
                  projs_on=np.array([], dtype=bool),
                  apply_proj=False,
                  remove_dc=True,  # for EOG/ECG
                  filter_coefs=None,
                  filter_bounds=None,
                  noise_cov=None,
                  # scalings
                  scalings=scalings,
                  units=units,
                  unit_scalings=unit_scalings,
                  # colors
                  ch_color_bad=bad_color,
                  ch_color_dict=color,
                  # display
                  butterfly=False,
                  clipping=None,
                  scrollbars_visible=show_scrollbars,
                  scalebars_visible=False,
                  window_title=title)
    if is_epo:
        params.update(n_epochs=n_epochs,
                      boundary_times=boundary_times,
                      event_id_rev=event_id_rev,
                      event_color_dict=event_color_dict,
                      event_nums=event_nums,
                      epoch_color_bad=(1, 0, 0),
                      epoch_colors=None,
                      xlabel='Epoch number')


    fig = _browse_figure(**params) # NOTE: Adapted this to include topos

    # MVE: define some colors for bad component labels
    import matplotlib.colors as mcolors
    c = list(mcolors.TABLEAU_COLORS.keys())
    idx = [c.index(i) for i in c if 'red' in i]
    for i in idx:
        del c[i]
    c = c[:len(bad_labels_list)+1] # keep as many as required.
    fig.mne.bad_label_colors = c

    fig._update_picks()

    # update data, and plot
    fig._update_trace_offsets()
    fig._update_data()
    fig._draw_traces() # NOTE: This one I updated to include topos

    # plot annotations (if any)
    if is_raw:
        fig._setup_annotation_colors()
        fig._update_annotation_segments()
        fig._draw_annotations()
        fig._alt_title() # MVE addition

    # for blitting
    fig.canvas.flush_events()
    fig.mne.bg = fig.canvas.copy_from_bbox(fig.bbox)

    plt_show(show, block=block)
    return fig

#%% ------------------------------------------------------------


from mne.viz._figure import MNEFigure, MNEBrowseFigure, _figure, _patched_canvas

import numpy as np
from mne.viz.utils import (plt_show, plot_sensors)
from mne.defaults import _handle_default
from mne.io.pick import pick_types

# CONSTANTS (inches)
ANNOTATION_FIG_PAD = 0.1
ANNOTATION_FIG_MIN_H = 2.9  # fixed part, not including radio buttons/labels
ANNOTATION_FIG_W = 5.0
ANNOTATION_FIG_CHECKBOX_COLUMN_W = 0.5

def _browse_figure(inst, **kwargs):
    """Instantiate a new MNE browse-style figure."""
    from mne.viz.utils import _get_figsize_from_config
    figsize = kwargs.pop('figsize', _get_figsize_from_config())
    fig = _figure(inst=inst, toolbar=False, FigureClass=osl_MNEBrowseFigure, # MVE: ONLY DIFFERENCE IS IT NOW POINTS TO MY VERSION OF THE FIGURE CLASS
             figsize=figsize, **kwargs)
    # initialize zen mode (can't do in __init__ due to get_position() calls)
    fig.canvas.draw()
    fig.mne.fig_size_px = fig._get_size_px()
    fig.mne.zen_w = (fig.mne.ax_vscroll.get_position().xmax -
                     fig.mne.ax_main.get_position().xmax)
    fig.mne.zen_h = (fig.mne.ax_main.get_position().ymin -
                     fig.mne.ax_hscroll.get_position().ymin)
    # if scrollbars are supposed to start hidden, set to True and then toggle
    if not fig.mne.scrollbars_visible:
        fig.mne.scrollbars_visible = True
        fig._toggle_scrollbars()
    # add event callbacks
    fig._add_default_callbacks()
    return fig

# MVE: MY VERSION OF THE MNEBROWSEFIGURE CLASS
class osl_MNEBrowseFigure(MNEBrowseFigure):
    """Interactive figure with scrollbars, for data browsing."""

    def __init__(self, inst, figsize, ica=None, xlabel='Time (s)', **kwargs):
        from matplotlib.colors import to_rgba_array
        from matplotlib.ticker import (FixedLocator, FixedFormatter,
                                       FuncFormatter, NullFormatter)
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button
        from matplotlib.transforms import blended_transform_factory
        from mpl_toolkits.axes_grid1.axes_size import Fixed
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from mne import BaseEpochs
        from mne.io import BaseRaw
        from mne.preprocessing import ICA
        import mne # MVE: ADDED FULL IMPORT OF MNE

        super().__init__(figsize=figsize, inst=inst, ica=ica, **kwargs)

        # what kind of data are we dealing with?
        if isinstance(ica, ICA):
            self.mne.instance_type = 'ica'
        elif isinstance(inst, BaseRaw):
            self.mne.instance_type = 'raw'
        elif isinstance(inst, BaseEpochs):
            self.mne.instance_type = 'epochs'
        else:
            raise TypeError('Expected an instance of Raw, Epochs, or ICA, '
                            f'got {type(inst)}.')
        self.mne.ica_type = None
        if self.mne.instance_type == 'ica':
            if isinstance(self.mne.ica_inst, BaseRaw):
                self.mne.ica_type = 'raw'
            elif isinstance(self.mne.ica_inst, BaseEpochs):
                self.mne.ica_type = 'epochs'
        self.mne.is_epochs = 'epochs' in (self.mne.instance_type,
                                          self.mne.ica_type)

        # things that always start the same
        self.mne.ch_start = 0
        self.mne.projector = None
        self.mne.projs_active = np.array([p['active'] for p in self.mne.projs])
        self.mne.whitened_ch_names = list()
        self.mne.use_noise_cov = self.mne.noise_cov is not None
        self.mne.zorder = dict(patch=0, grid=1, ann=2, events=3, bads=4,
                               data=5, mag=6, grad=7, scalebar=8, vline=9)
        # additional params for epochs (won't affect raw / ICA)
        self.mne.epoch_traces = list()
        self.mne.bad_epochs = list()
        # annotations
        self.mne.annotations = list()
        self.mne.hscroll_annotations = list()
        self.mne.annotation_segments = list()
        self.mne.annotation_texts = list()
        self.mne.new_annotation_labels = list()
        self.mne.annotation_segment_colors = dict()
        self.mne.annotation_hover_line = None
        self.mne.draggable_annotations = False
        # lines
        self.mne.event_lines = None
        self.mne.event_texts = list()
        self.mne.vline_visible = False
        # scalings
        self.mne.scale_factor = 0.5 if self.mne.butterfly else 1.
        self.mne.scalebars = dict()
        self.mne.scalebar_texts = dict()
        # ancillary child figures
        self.mne.child_figs = list()
        self.mne.fig_help = None
        self.mne.fig_proj = None
        self.mne.fig_histogram = None
        self.mne.fig_selection = None
        self.mne.fig_annotation = None
        # MVE: bad component labels
        self.mne.bad_labels_list = kwargs['bad_labels_list']

        # MAIN AXES: default sizes (inches)
        # XXX simpler with constrained_layout? (when it's no longer "beta")
        l_margin = 1.
        r_margin = 0.1
        b_margin = 0.45
        t_margin = 0.35
        scroll_width = 0.25
        hscroll_dist = 0.25
        vscroll_dist = 0.1
        help_width = scroll_width * 2
        # MVE: ADD SIZES FOR TOPOS
        n_topos = len(np.unique([mne.io.pick.channel_type(ica.info,ch) for ch in mne.pick_types(ica.info,meg=True)]))
        topo_width_ratio = 8+n_topos #1
        topo_dist = self._inch_to_rel(0.05) #0.25

        # MAIN AXES: default margins (figure-relative coordinates)
        self.canvas.figure.clear() # clear axes (inherited from MNE)
        left = self._inch_to_rel(l_margin - vscroll_dist - help_width)
        right = 1 - self._inch_to_rel(r_margin)
        bottom = self._inch_to_rel(b_margin, horiz=False)
        top = 1 - self._inch_to_rel(t_margin, horiz=False)
        height = top - bottom

        # MVE: ADAPT SIZES OF TIME COURSE SUBPLOT AND ADD TOPO PLOT SIZE
        fullwidth = right-left
        width = (topo_width_ratio - n_topos) * (fullwidth - n_topos*topo_dist) / topo_width_ratio - (self._inch_to_rel(hscroll_dist)+self._inch_to_rel(scroll_width))  # width = right - left
        topo_width = (fullwidth-topo_dist)/topo_width_ratio
        topo_height = (height-self._inch_to_rel(hscroll_dist+b_margin))/self.mne.n_channels-topo_dist

        position = [left + n_topos*(topo_width + topo_dist), bottom, width, height] #position = [left, bottom, width, height]
        # Main axes must be a subplot for subplots_adjust to work (so user can
        # adjust margins). That's why we don't use the Divider class directly.
        ax_main = self.add_axes(position) #ax_main = self.add_subplot(1, 1, 1, position=position) # MVE: USE ADD_AXES INSTEAD OF ADD_SUBPLOT
        # MVE: CREATE TOPO AXES
        ax_topo=np.empty((n_topos,self.mne.n_channels),dtype=object)
        for i in np.arange(n_topos):
            for j in np.arange(self.mne.n_channels):
                topo_position = [left + i * (topo_width + topo_dist), bottom + self._inch_to_rel(hscroll_dist+b_margin) + ((self.mne.n_channels-1)-j)*(topo_height+topo_dist)+topo_dist, topo_width, topo_height]
                ax_topo[i,j] = self.add_axes(topo_position)
                ax_topo[i,j].set_axis_off()

        self.subplotpars.update(left=left, bottom=bottom, top=top, right=right)
        div = make_axes_locatable(ax_main)
        # this only gets shown in zen mode
        self.mne.zen_xlabel = ax_main.set_xlabel(xlabel)
        self.mne.zen_xlabel.set_visible(not self.mne.scrollbars_visible)

        # SCROLLBARS
        ax_hscroll = div.append_axes(position='bottom',
                                     size=Fixed(scroll_width),
                                     pad=Fixed(hscroll_dist))
        # TODO: MVE: MAKE MANUAL SIZED SCROLLBAR. BELOW ADD ^ AND V BUTTONS TO JUMP 10 COMPONENTS AT THE TIME.
        #vscroll_bottom = 0.7
        #ax_vscroll = self.add_axes([right-self._inch_to_rel(vscroll_dist), bottom+self._inch_to_rel(vscroll_bottom), scroll_width, height-self._inch_to_rel(vscroll_bottom)])
        ax_vscroll = div.append_axes(position='right',
                                     size=Fixed(scroll_width),
                                     pad=Fixed(vscroll_dist))
        ax_hscroll.get_yaxis().set_visible(False)
        ax_hscroll.set_xlabel(xlabel)
        ax_vscroll.set_axis_off()
        # TODO: MVE: ^ AND V BUTTONS
        # ax_vscroll_up = self.add_axes([right-self._inch_to_rel(vscroll_dist), bottom+self._inch_to_rel(vscroll_bottom/2), scroll_width, height-self._inch_to_rel(vscroll_bottom/2-0.05)])
        # ax_vscroll_down = self.add_axes([right-self._inch_to_rel(vscroll_dist), bottom, scroll_width, height-self._inch_to_rel(vscroll_bottom/2-0.05)])
        # with _patched_canvas(ax_vscroll_up.figure):
        #     self.mne.button_help = Button(ax_vscroll_up, '^')

        # HORIZONTAL SCROLLBAR PATCHES (FOR MARKING BAD EPOCHS)
        if self.mne.is_epochs:
            epoch_nums = self.mne.inst.selection
            for ix, _ in enumerate(epoch_nums):
                start = self.mne.boundary_times[ix]
                width = np.diff(self.mne.boundary_times[ix:ix + 2])[0]
                ax_hscroll.add_patch(
                    Rectangle((start, 0), width, 1, color='none',
                              zorder=self.mne.zorder['patch']))
            # add epoch boundaries & center epoch numbers between boundaries
            midpoints = np.convolve(self.mne.boundary_times, np.ones(2),
                                    mode='valid') / 2
            # both axes, major ticks: gridlines
            for _ax in (ax_main, ax_hscroll):
                _ax.xaxis.set_major_locator(
                    FixedLocator(self.mne.boundary_times[1:-1]))
                _ax.xaxis.set_major_formatter(NullFormatter())
            grid_kwargs = dict(color=self.mne.fgcolor, axis='x',
                               zorder=self.mne.zorder['grid'])
            ax_main.grid(linewidth=2, linestyle='dashed', **grid_kwargs)
            ax_hscroll.grid(alpha=0.5, linewidth=0.5, linestyle='solid',
                            **grid_kwargs)
            # main axes, minor ticks: ticklabel (epoch number) for every epoch
            ax_main.xaxis.set_minor_locator(FixedLocator(midpoints))
            ax_main.xaxis.set_minor_formatter(FixedFormatter(epoch_nums))
            # hscroll axes, minor ticks: up to 20 ticklabels (epoch numbers)
            ax_hscroll.xaxis.set_minor_locator(
                FixedLocator(midpoints, nbins=20))
            ax_hscroll.xaxis.set_minor_formatter(
                FuncFormatter(lambda x, pos: self._get_epoch_num_from_time(x)))
            # hide some ticks
            ax_main.tick_params(axis='x', which='major', bottom=False)
            ax_hscroll.tick_params(axis='x', which='both', bottom=False)

        # VERTICAL SCROLLBAR PATCHES (COLORED BY CHANNEL TYPE)
        ch_order = self.mne.ch_order
        for ix, pick in enumerate(ch_order):
            this_color = (self.mne.ch_color_bad
                          if self.mne.ch_names[pick] in self.mne.info['bads']
                          else self.mne.ch_color_dict)
            if isinstance(this_color, dict):
                this_color = this_color[self.mne.ch_types[pick]]
            ax_vscroll.add_patch(
                Rectangle((0, ix), 1, 1, color=this_color,
                          zorder=self.mne.zorder['patch']))
        ax_vscroll.set_ylim(len(ch_order), 0)
        ax_vscroll.set_visible(not self.mne.butterfly)
        # SCROLLBAR VISIBLE SELECTION PATCHES
        sel_kwargs = dict(alpha=0.3, linewidth=4, clip_on=False,
                          edgecolor=self.mne.fgcolor)
        vsel_patch = Rectangle((0, 0), 1, self.mne.n_channels,
                               facecolor=self.mne.bgcolor, **sel_kwargs)
        ax_vscroll.add_patch(vsel_patch)
        hsel_facecolor = np.average(
            np.vstack((to_rgba_array(self.mne.fgcolor),
                       to_rgba_array(self.mne.bgcolor))),
            axis=0, weights=(3, 1))  # 75% foreground, 25% background
        hsel_patch = Rectangle((self.mne.t_start, 0), self.mne.duration, 1,
                               facecolor=hsel_facecolor, **sel_kwargs)
        ax_hscroll.add_patch(hsel_patch)
        ax_hscroll.set_xlim(self.mne.first_time, self.mne.first_time +
                            self.mne.n_times / self.mne.info['sfreq'])
        # VLINE
        vline_color = (0., 0.75, 0.)
        vline_kwargs = dict(visible=False, animated=True,
                            zorder=self.mne.zorder['vline'])
        if self.mne.is_epochs:
            x = np.arange(self.mne.n_epochs)
            vline = ax_main.vlines(
                x, 0, 1, colors=vline_color, **vline_kwargs)
            vline.set_transform(blended_transform_factory(ax_main.transData,
                                                          ax_main.transAxes))
            vline_hscroll = None
        else:
            vline = ax_main.axvline(0, color=vline_color, **vline_kwargs)
            vline_hscroll = ax_hscroll.axvline(0, color=vline_color,
                                               **vline_kwargs)
        vline_text = ax_hscroll.text(
            self.mne.first_time, 1.2, '', fontsize=10, ha='right', va='bottom',
            color=vline_color, **vline_kwargs)

        # HELP BUTTON: initialize in the wrong spot...
        ax_help = div.append_axes(position='left',
                                  size=Fixed(help_width),
                                  pad=Fixed(vscroll_dist))
        # HELP BUTTON: ...move it down by changing its locator
        loc = div.new_locator(nx=0, ny=0)
        ax_help.set_axes_locator(loc)
        # HELP BUTTON: make it a proper button
        with _patched_canvas(ax_help.figure):
            self.mne.button_help = Button(ax_help, 'Help')
        # PROJ BUTTON
        ax_proj = None
        if len(self.mne.projs) and not inst.proj:
            proj_button_pos = [
                1 - self._inch_to_rel(r_margin + scroll_width),  # left
                self._inch_to_rel(b_margin, horiz=False),        # bottom
                self._inch_to_rel(scroll_width),                 # width
                self._inch_to_rel(scroll_width, horiz=False)     # height
            ]
            loc = div.new_locator(nx=4, ny=0)
            ax_proj = self.add_axes(proj_button_pos)
            ax_proj.set_axes_locator(loc)
            with _patched_canvas(ax_help.figure):
                self.mne.button_proj = Button(ax_proj, 'Prj')

        # INIT TRACES
        self.mne.trace_kwargs = dict(antialiased=True, linewidth=0.5)
        self.mne.traces = ax_main.plot(
            np.full((1, self.mne.n_channels), np.nan), **self.mne.trace_kwargs)

        # MVE: INITIALLY THIS IS WHERE I INITIALIZED THE TOPOS. TURNS OUT ITS REDUNDANT BECAUSE IT IS TAKEN CARE OF IN
        # THE INTERACTIVE PART OF THE FIGURE. IT ALSO SOLVES THE EXTRA BONUS FIGURE
        # INIT TOPOS
        # NOTE: Commenting the next line out seems to not break the code, but to solve the bonus figure that is created
        # upon running the code.
        # self.plot_topos(ica, ax_topo, self.mne.picks[:self.mne.n_channels])

        # SAVE UI ELEMENT HANDLES
        vars(self.mne).update(
            ax_main=ax_main, ax_help=ax_help, ax_proj=ax_proj,
            ax_hscroll=ax_hscroll, ax_vscroll=ax_vscroll,
            vsel_patch=vsel_patch, hsel_patch=hsel_patch, vline=vline,
            vline_hscroll=vline_hscroll, vline_text=vline_text)

    def _draw_traces(self):
        from mne import pick_types
        from mne.io.pick import channel_type
        """Draw (or redraw) the channel data."""
        from matplotlib.colors import to_rgba_array
        from matplotlib.patches import Rectangle
        # clear scalebars
        if self.mne.scalebars_visible:
            self._hide_scalebars()
        # get info about currently visible channels
        picks = self.mne.picks
        ch_names = self.mne.ch_names[picks]
        ch_types = self.mne.ch_types[picks]
        bad_bool = np.in1d(ch_names, self.mne.info['bads'])
        # MVE addition
        bad_int=[]
        for i in range(len(ch_names)):
            if ch_names[i] in self.mne.info['bads']:
                if ch_names[i] in list(self.mne.ica.labels_.keys()):
                    bad_int.append(self.mne.bad_labels_list.index(self.mne.ica.labels_[ch_names[i]])+2)
                else:
                    bad_int.append(0)
            else:
                if ch_names[i] in list(self.mne.ica.labels_.keys()): # remove entry
                    del self.mne.ica.labels_[ch_names[i]]
                bad_int.append(np.nan)

        good_ch_colors = [self.mne.ch_color_dict[_type] for _type in ch_types]
        # Now match colors to specific artifact labels
        c = [self.mne.ch_color_bad] + self.mne.bad_label_colors
        ch_colors = to_rgba_array(
            [c[_bad] if _bad is not np.nan else _color
             for _bad, _color in zip(bad_int, good_ch_colors)])

        self.mne.ch_colors = np.array(good_ch_colors)  # use for unmarking bads
        labels = self.mne.ax_main.yaxis.get_ticklabels()
        if self.mne.butterfly:
            for label in labels:
                label.set_color(self.mne.fgcolor)
        else:
            for label, color in zip(labels, ch_colors):
                label.set_color(color)
        # decim
        decim = np.ones_like(picks)
        data_picks_mask = np.in1d(picks, self.mne.picks_data)
        decim[data_picks_mask] = self.mne.decim
        # decim can vary by channel type, so compute different `times` vectors
        decim_times = {decim_value:
                       self.mne.times[::decim_value] + self.mne.first_time
                       for decim_value in set(decim)}
        # add more traces if needed
        n_picks = len(picks)
        if n_picks > len(self.mne.traces):
            n_new_chs = n_picks - len(self.mne.traces)
            new_traces = self.mne.ax_main.plot(np.full((1, n_new_chs), np.nan),
                                               **self.mne.trace_kwargs)
            self.mne.traces.extend(new_traces)
        # remove extra traces if needed
        extra_traces = self.mne.traces[n_picks:]
        for trace in extra_traces:
            self.mne.ax_main.lines.remove(trace)
        self.mne.traces = self.mne.traces[:n_picks]

        # check for bad epochs
        time_range = (self.mne.times + self.mne.first_time)[[0, -1]]
        if self.mne.instance_type == 'epochs':
            epoch_ix = np.searchsorted(self.mne.boundary_times, time_range)
            epoch_ix = np.arange(epoch_ix[0], epoch_ix[1])
            epoch_nums = self.mne.inst.selection[epoch_ix[0]:epoch_ix[-1] + 1]
            visible_bad_epochs = epoch_nums[
                np.in1d(epoch_nums, self.mne.bad_epochs).nonzero()]
            while len(self.mne.epoch_traces):
                _trace = self.mne.epoch_traces.pop(-1)
                self.mne.ax_main.lines.remove(_trace)
            # handle custom epoch colors (for autoreject integration)
            if self.mne.epoch_colors is None:
                # shape: n_traces × RGBA → n_traces × n_epochs × RGBA
                custom_colors = np.tile(ch_colors[:, None, :],
                                        (1, self.mne.n_epochs, 1))
            else:
                custom_colors = np.empty((len(self.mne.picks),
                                          self.mne.n_epochs, 4))
                for ii, _epoch_ix in enumerate(epoch_ix):
                    this_colors = self.mne.epoch_colors[_epoch_ix]
                    custom_colors[:, ii] = to_rgba_array([this_colors[_ch]
                                                          for _ch in picks])
            # override custom color on bad epochs
            for _bad in visible_bad_epochs:
                _ix = epoch_nums.tolist().index(_bad)
                _cols = np.array([self.mne.epoch_color_bad,
                                  self.mne.ch_color_bad])[bad_bool.astype(int)]
                custom_colors[:, _ix] = to_rgba_array(_cols)

        # update traces
        ylim = self.mne.ax_main.get_ylim()
        for ii, line in enumerate(self.mne.traces):
            this_name = ch_names[ii]
            this_type = ch_types[ii]
            this_offset = self.mne.trace_offsets[ii]
            this_times = decim_times[decim[ii]]
            this_data = this_offset - self.mne.data[ii] * self.mne.scale_factor
            this_data = this_data[..., ::decim[ii]]
            # clip
            if self.mne.clipping == 'clamp':
                this_data = np.clip(this_data, -0.5, 0.5)
            elif self.mne.clipping is not None:
                clip = self.mne.clipping * (0.2 if self.mne.butterfly else 1)
                bottom = max(this_offset - clip, ylim[1])
                height = min(2 * clip, ylim[0] - bottom)
                rect = Rectangle(xy=np.array([time_range[0], bottom]),
                                 width=time_range[1] - time_range[0],
                                 height=height,
                                 transform=self.mne.ax_main.transData)
                line.set_clip_path(rect)
            # prep z order
            is_bad_ch = this_name in self.mne.info['bads']
            this_z = self.mne.zorder['bads' if is_bad_ch else 'data']
            if self.mne.butterfly and not is_bad_ch:
                this_z = self.mne.zorder.get(this_type, this_z)
            # plot each trace multiple times to get the desired epoch coloring.
            # use masked arrays to plot discontinuous epochs that have the same
            # color in a single plot() call.
            if self.mne.instance_type == 'epochs':
                this_colors = custom_colors[ii]
                for cix, color in enumerate(np.unique(this_colors, axis=0)):
                    bool_ixs = (this_colors == color).all(axis=1)
                    mask = np.zeros_like(this_times, dtype=bool)
                    _starts = self.mne.boundary_times[epoch_ix][bool_ixs]
                    _stops = self.mne.boundary_times[epoch_ix + 1][bool_ixs]
                    for _start, _stop in zip(_starts, _stops):
                        _mask = np.logical_and(_start < this_times,
                                               this_times <= _stop)
                        mask = mask | _mask
                    _times = np.ma.masked_array(this_times, mask=~mask)
                    # always use the existing traces first
                    if cix == 0:
                        line.set_xdata(_times)
                        line.set_ydata(this_data)
                        line.set_color(color)
                        line.set_zorder(this_z)
                    else:  # make new traces as needed
                        _trace = self.mne.ax_main.plot(
                            _times, this_data, color=color, zorder=this_z,
                            **self.mne.trace_kwargs)
                        self.mne.epoch_traces.extend(_trace)
            else:
                line.set_xdata(this_times)
                line.set_ydata(this_data)
                line.set_color(ch_colors[ii])
                line.set_zorder(this_z)
        # update xlim
        self.mne.ax_main.set_xlim(*time_range)
        # draw scalebars maybe
        if self.mne.scalebars_visible:
            self._show_scalebars()
        # redraw event lines
        if self.mne.event_times is not None:
            self._draw_event_lines()

        # MVE: ADDITION FOR TOPOS:
        n_topos = len(picks)
        n_chtype=len(np.unique([channel_type(self.mne.ica.info, ch) for ch in pick_types(self.mne.ica.info, meg=True)]))
        ax_topo = np.reshape(self.get_axes()[1:n_topos * n_chtype + 1], (n_chtype, n_topos))
        self.plot_topos(self.mne.ica, ax_topo, self.mne.picks)

        # MVE: TITLE STUFF
        self._alt_title()

    def plot_topos(self, ica, ax_topo, picks): # MVE: ADDITION FOR TOPOS
        import numpy as np
        import mne
        n_topos = len(picks)
        ica_tmp = ica.copy()
        ica_tmp._ica_names = ['' for i in ica_tmp._ica_names]
        chtype = np.unique([mne.io.pick.channel_type(ica.info, ch) for ch in mne.pick_types(ica.info, meg=True)])
        n_chtype = len(chtype)
        for i in range(n_chtype):
            for j in range(n_topos):
                mne.viz.ica._plot_ica_topomap(ica_tmp, idx=picks[j], ch_type=chtype[i], axes=ax_topo[i, j],
                                              vmin=None, vmax=None, cmap='RdBu_r', colorbar=False,
                                              title=None, show=True, outlines='head', contours=0,
                                              image_interp='bilinear', res=64,
                                              sensors=False, allow_ref_meg=False,
                                              sphere=None
                                              )
            ax_topo[i, 0].set_title(f'{chtype[i]}')

    def _close(self, event):
        """Handle close events (via keypress or window [x])."""
        from matplotlib.pyplot import close
        from mne.utils import logger, set_config
        # write out bad epochs (after converting epoch numbers to indices)
        if self.mne.instance_type == 'epochs':
            bad_ixs = np.in1d(self.mne.inst.selection,
                              self.mne.bad_epochs).nonzero()[0]
            self.mne.inst.drop(bad_ixs)
        # write bad channels back to instance (don't do this for proj;
        # proj checkboxes are for viz only and shouldn't modify the instance)
        if self.mne.instance_type in ('raw', 'epochs'):
            self.mne.inst.info['bads'] = self.mne.info['bads']
            logger.info(
                f"Channels marked as bad: {self.mne.info['bads'] or 'none'}")
        # ICA excludes
        elif self.mne.instance_type == 'ica':
            self.mne.ica.exclude = [self.mne.ica._ica_names.index(ch)
                                    for ch in self.mne.info['bads']]
            # MVE: remove bad component labels that were reversed to good component
            tmp=self.mne.ica.labels_.keys()
            for ch in tmp:
                if ch not in self.mne.info['bads']:
                    del self.mne.ica.labels_[ch]
            # label bad components without a manual label as "unknown"
            for ch in self.mne.info['bads']:
                if ch not in self.mne.ica.labels_:
                    self.mne.ica.labels_[ch] = 'unknown'
        # write window size to config
        size = ','.join(self.get_size_inches().astype(str))
        set_config('MNE_BROWSE_RAW_SIZE', size, set_env=False)
        # Clean up child figures (don't pop(), child figs remove themselves)
        while len(self.mne.child_figs):
            fig = self.mne.child_figs[-1]
            close(fig)

    def _keypress(self, event):
        from mne.viz.utils import _events_off
        """Handle keypress events."""
        key = event.key
        n_channels = self.mne.n_channels
        if self.mne.is_epochs:
            last_time = self.mne.n_times / self.mne.info['sfreq']
        else:
            last_time = self.mne.inst.times[-1]
        # scroll up/down
        if key in ('down', 'up'):
            direction = -1 if key == 'up' else 1
            # butterfly case
            if self.mne.butterfly:
                return
            # group_by case
            elif self.mne.fig_selection is not None:
                buttons = self.mne.fig_selection.mne.radio_ax.buttons
                labels = [label.get_text() for label in buttons.labels]
                current_label = buttons.value_selected
                current_idx = labels.index(current_label)
                selections_dict = self.mne.ch_selections
                penult = current_idx < (len(labels) - 1)
                pre_penult = current_idx < (len(labels) - 2)
                has_custom = selections_dict.get('Custom', None) is not None
                def_custom = len(selections_dict.get('Custom', list()))
                up_ok = key == 'up' and current_idx > 0
                down_ok = key == 'down' and (
                    pre_penult or
                    (penult and not has_custom) or
                    (penult and has_custom and def_custom))
                if up_ok or down_ok:
                    buttons.set_active(current_idx + direction)
            # normal case
            else:
                ceiling = len(self.mne.ch_order) - n_channels
                ch_start = self.mne.ch_start + direction * n_channels
                self.mne.ch_start = np.clip(ch_start, 0, ceiling)
                self._update_picks()
                self._update_vscroll()
                self._redraw()
        # scroll left/right
        elif key in ('right', 'left', 'shift+right', 'shift+left'):
            old_t_start = self.mne.t_start
            direction = 1 if key.endswith('right') else -1
            if self.mne.is_epochs:
                denom = 1 if key.startswith('shift') else self.mne.n_epochs
            else:
                denom = 1 if key.startswith('shift') else 4
            t_max = last_time - self.mne.duration
            t_start = self.mne.t_start + direction * self.mne.duration / denom
            self.mne.t_start = np.clip(t_start, self.mne.first_time, t_max)
            if self.mne.t_start != old_t_start:
                self._update_hscroll()
                self._redraw(annotations=True)
        # scale traces
        elif key in ('=', '+', '-'):
            scaler = 1 / 1.1 if key == '-' else 1.1
            self.mne.scale_factor *= scaler
            self._redraw(update_data=False)
        # change number of visible channels
        elif (key in ('pageup', 'pagedown') and
              self.mne.fig_selection is None and
              not self.mne.butterfly):
            new_n_ch = n_channels + (1 if key == 'pageup' else -1)
            self.mne.n_channels = np.clip(new_n_ch, 1, len(self.mne.ch_order))
            # add new chs from above if we're at the bottom of the scrollbar
            ch_end = self.mne.ch_start + self.mne.n_channels
            if ch_end > len(self.mne.ch_order) and self.mne.ch_start > 0:
                self.mne.ch_start -= 1
                self._update_vscroll()
            # redraw only if changed
            if self.mne.n_channels != n_channels:
                self._update_picks()
                self._update_trace_offsets()
                self._redraw(annotations=True)
        # change duration
        elif key in ('home', 'end'):
            dur_delta = 1 if key == 'end' else -1
            if self.mne.is_epochs:
                self.mne.n_epochs = np.clip(self.mne.n_epochs + dur_delta,
                                            1, len(self.mne.inst))
                min_dur = len(self.mne.inst.times) / self.mne.info['sfreq']
                dur_delta *= min_dur
            else:
                min_dur = 3 * np.diff(self.mne.inst.times[:2])[0]
            old_dur = self.mne.duration
            new_dur = self.mne.duration + dur_delta
            self.mne.duration = np.clip(new_dur, min_dur, last_time)
            if self.mne.duration != old_dur:
                if self.mne.t_start + self.mne.duration > last_time:
                    self.mne.t_start = last_time - self.mne.duration
                self._update_hscroll()
                if key == 'end' and self.mne.vline_visible:  # prevent flicker
                    self._show_vline(None)
                self._redraw()
        elif key == '?':  # help window
            self._toggle_help_fig(event)
        elif key == 'a':  # annotation mode
            self._toggle_annotation_fig()
        elif key == 'b' and self.mne.instance_type != 'ica':  # butterfly mode
            self._toggle_butterfly()
        elif key == 'd':  # DC shift
            self.mne.remove_dc = not self.mne.remove_dc
            self._redraw()
        elif key == 'h' and self.mne.instance_type == 'epochs':  # histogram
            self._toggle_epoch_histogram()
        elif key == 'j' and len(self.mne.projs):  # SSP window
            self._toggle_proj_fig()
        elif key == 'p':  # toggle draggable annotations
            self._toggle_draggable_annotations(event)
            if self.mne.fig_annotation is not None:
                checkbox = self.mne.fig_annotation.mne.drag_checkbox
                with _events_off(checkbox):
                    checkbox.set_active(0)
        elif key == 's':  # scalebars
            self._toggle_scalebars(event)
        elif key == 'w':  # toggle noise cov whitening
            if self.mne.noise_cov is not None:
                self.mne.use_noise_cov = not self.mne.use_noise_cov
                self._update_projector()
                self._update_yaxis_labels()  # add/remove italics
                self._redraw()
        elif key == 'z':  # zen mode: hide scrollbars and buttons
            self._toggle_scrollbars()
            self._redraw(update_data=False)
        elif int(key) in range(len(self.mne.bad_labels_list)+1): # MVE addition for labeling artifact type of bad components
            if len(self.mne.info['bads'])>0:
                last_bad_component = self.mne.info['bads'][-1]
                # save bad component label in dict.
                self.mne.ica.labels_[last_bad_component]=self.mne.bad_labels_list[int(key)-1]
                print(f'Component {last_bad_component} labeled as "{self.mne.bad_labels_list[int(key) - 1]}"')
                self._draw_traces() # MVE: This makes sure the traces are given the corresponding color right away
        else:  # check for close key / fullscreen toggle
            super()._keypress(event)

    def _alt_title(self):
        self.mne.ax_main.texts.clear()
        x = np.arange(self.mne.ax_main.get_xlim()[0], self.mne.ax_main.get_xlim()[1], (self.mne.ax_main.get_xlim()[1]-self.mne.ax_main.get_xlim()[0])/(len(self.mne.bad_labels_list)+2))
        for i in range(len(self.mne.bad_labels_list)+2):
            y = self.mne.ax_main.get_ylim()[1]
            if i==0:
                text = self.mne.ax_main.annotate('bad_segments', (x[i],y), xytext=(0, 9),
                            textcoords='offset points', ha='center',
                            va='baseline', color='red')
            elif i == len(self.mne.bad_labels_list)+1:
                    text = self.mne.ax_main.annotate('unknown', (x[i], y), xytext=(0, 9),
                                                     textcoords='offset points', ha='center',
                                                     va='baseline', color='gray')
            else:
                text = self.mne.ax_main.annotate(f'{i}: {self.mne.bad_labels_list[i-1]}', (x[i],y), xytext=(0, 9),
                            textcoords='offset points', ha='center',
                            va='baseline', color=self.mne.bad_label_colors[i])

            self.mne.annotation_texts.append(text)
