
import pickle
from copy import deepcopy
from pathlib import Path

import glmtools as glm
import matplotlib.pyplot as plt
import mne
import numpy as np
from sails.stft import glm_periodogram
from scipy import signal, stats
from .glm_base import GLMBaseResult, GroupGLMBaseResult, SensorClusterPerm

from matplotlib.patches import ConnectionPatch

#%% ---------------------------------------
#
# GLM-Spectrum classes designed to work with GLM-Spectra computed from  MNE
# format sensorspace data

class SensorGLMSpectrum(GLMBaseResult):
    """A class for GLM-Spectra fitted from MNE-Python Raw objects."""

    def __init__(self, glmsp, info):

        self.f = glmsp.f
        self.config = glmsp.config
        super().__init__(glmsp.model, glmsp.design, info, data=glmsp.data)

    def plot_joint_spectrum(self, contrast=0, freqs='auto', base=1, ax=None,
                            topo_scale='joint', lw=0.5,  ylabel=None, title=None,
                            ylim=None, xtick_skip=1, topo_prop=1/5, metric='copes'):
        """Plot a GLM-Spectrum contrast with spatial line colouring and topograpies.

        Parameters
        ----------
        contrast : int
             Contrast to plot (Default value = 0)
        freqs : {list, tuple or 'auto'}
             Which frequencies to plot topos for (Default value = 'auto')
        base : float
             The x-axis scaling, set to 0.5 for sqrt freq axis (Default value = 1)
        ax : {None or axis handle}
             Axis to plot into (Default value = None)
        topo_scale : {'joint' or None}
             Whether to fix topomap colour scales across all topos ('joint') or
             leave them individual (Default value = 'joint')
        lw : flot
             Line width(Default value = 0.5)
        ylabel : str
             Y-axis label(Default value = 'Power')
        title : str
             Plot title(Default value = None)
        ylim : {tuple or list}
             min and max values for y-axis (Default value = None)
        xtick_skip : int
             Number of xaxis ticks to skip, useful for tight plots (Default value = 1)
        topo_prop : float
             Proportion of plot dedicted to topomaps(Default value = 1/3)
        metric : {'copes' or 'tstats}
             Which metric to plot? (Default value = 'copes')

        """
        if metric == 'copes':
            spec = self.model.copes[contrast, :, :].T
            ylabel = 'Power' if ylabel is None else ylabel
        elif metric == 'varcopes':
            spec = self.model.varcopes[contrast, :, :].T
            ylabel = 'Standard-Error' if ylabel is None else ylabel
        elif metric == 'tstats':
            spec = self.model.tstats[contrast, :, :].T
            ylabel = 't-statistics' if ylabel is None else ylabel
        else:
            raise ValueError("Metric '{}' not recognised".format(metric))

        if title is None:
            title = 'C {} : {}'.format(contrast, self.design.contrast_names[contrast])

        plot_joint_spectrum(self.f, spec, self.info, freqs=freqs, base=base,
                            topo_scale=topo_scale, lw=lw, ylabel=ylabel, title=title,
                            ylim=ylim, xtick_skip=xtick_skip, topo_prop=topo_prop, ax=ax)

    def plot_sensor_spectrum(self, contrast, sensor_proj=False,
                             xticks=None, xticklabels=None, lw=0.5, ax=None, title=None,
                             sensor_cols=True, base=1, ylabel=None, xtick_skip=1, metric='copes'):
        """Plot a GLM-Spectrum contrast with spatial line colouring.

        Parameters
        ----------
        contrast : int
            Contrast to plot
        sensor_proj : bool
            Whether to plot an inset topo showing spatial colours (Default
            value = False)
        xticks : array_like of float
            xtick positions (Default value = None)
        xticklabels : array_like of str
            labels for xticks (Default value = None)
        ax : {None or axis handle}
            Axis to plot into (Default value = None)
        lw : flot
            Line width(Default value = 0.5)
        ylabel : str
            Y-axis label(Default value = 'Power')
        title : str
            Plot title(Default value = None)
        ylim : {tuple or list}
            min and max values for y-axis (Default value = None)
        xtick_skip : int
            Number of xaxis ticks to skip, useful for tight plots (Default value = 1)
        topo_prop : float
            Proportion of plot dedicted to topomaps(Default value = 1/3)
        metric : {'copes' or 'tstats}
            Which metric to plot(Default value = 'copes')

        """
        if metric == 'copes':
            spec = self.model.copes[contrast, :, :].T
        elif metric == 'tstats':
            spec = self.model.tstats[contrast, :, :].T

        if title is None:
            title = 'C {} : {}'.format(contrast, self.design.contrast_names[contrast])

        plot_sensor_spectrum(self.f, spec, self.info, ax=ax, sensor_proj=sensor_proj,
                             xticks=xticks, xticklabels=xticklabels, lw=lw, title=title,
                             sensor_cols=sensor_cols, base=base, ylabel=ylabel, xtick_skip=xtick_skip)


class GroupSensorGLMSpectrum(GroupGLMBaseResult):
    """A class for group level GLM-Spectra fitted across mmultiple first-level
    GLM-Spectra computed from MNE-Python Raw objects"""

    def __init__(self, model, design, config, info, fl_contrast_names=None, data=None):

        self.f = config.freqvals
        super().__init__(model, design, info, fl_contrast_names=fl_contrast_names, data=data)

    def __str__(self):
        msg = 'GroupSensorGLMSpectrum\n'
        line = '\tData - {} Inputs, {} Channels and {} Frequencies\n'
        msg += line.format(self.design.design_matrix.shape[0], self.model.copes.shape[2], self.model.copes.shape[3])

        line = '\tFirst-Level - {} Regressors and {} Contrasts\n'
        msg += line.format(self.design.design_matrix.shape[1], self.model.copes.shape[1])

        line = '\tGroup-Level - {} Regressors and {} Contrasts\n'
        msg += line.format(self.design.design_matrix.shape[1], self.model.copes.shape[0])
        return msg

    def save_pkl(self, outname, overwrite=True, save_data=False):
        """Save GLM-Spectrum result to a pickle file.

        Parameters
        ----------
        outname : str
             Filename or full file path to write pickle to
        overwrite : bool
             Overwrite previous file if one exists? (Default value = True)
        save_data : bool
             Save STFT data in pickle? This is omitted by default to save disk
             space (Default value = False)

        """
        if Path(outname).exists() and not overwrite:
            msg = "{} already exists. Please delete or do use overwrite=True."
            raise ValueError(msg.format(outname))

        self.config.detrend_func = None  # Have to drop this to pickle

        # This is hacky - but pickles are all or nothing and I don't know how
        # else to do it. HDF5 would be better longer term
        if save_data == False:
            # Temporarily remove data before saving
            dd = self.data
            self.data = None

        with open(outname, 'bw') as outp:
            pickle.dump(self, outp)

        # Put data back
        if save_data == False:
            self.data = dd

    def plot_joint_spectrum(self, gcontrast=0, fcontrast=0, freqs='auto', base=1, ax=None,
                            topo_scale='joint', lw=0.5,  ylabel='Power', title=None,
                            ylim=None, xtick_skip=1, topo_prop=1/5, metric='copes'):
        """

        Parameters
        ----------
        gcontrast : int
             Group level contrast to plot (Default value = 0)
        fcontrast : int
             First level contrast to plot (Default value = 0)
        freqs : {list, tuple or 'auto'}
             Which frequencies to plot topos for (Default value = 'auto')
        base : float
             The x-axis scaling, set to 0.5 for sqrt freq axis (Default value = 1)
        ax : {None or axis handle}
             Axis to plot into (Default value = None)
        topo_scale : {'joint' or None}
             Whether to fix topomap colour scales across all topos ('joint') or
             leave them individual (Default value = 'joint')
        lw : flot
             Line width(Default value = 0.5)
        ylabel : str
             Y-axis label(Default value = 'Power')
        title : str
             Plot title(Default value = None)
        ylim : {tuple or list}
             min and max values for y-axis (Default value = None)
        xtick_skip : int
             Number of xaxis ticks to skip, useful for tight plots (Default value = 1)
        topo_prop : float
             Proportion of plot dedicted to topomaps(Default value = 1/3)
        metric : {'copes' or 'tstats}
             Which metric to plot(Default value = 'copes')

        """
        if metric == 'copes':
            spec = self.model.copes[gcontrast, fcontrast, :, :].T
        elif metric == 'varcopes':
            spec = self.model.varcopes[gcontrast, fcontrast, :, :].T
            ylabel = 'Standard-Error' if ylabel is None else ylabel
        elif metric == 'tstats':
            spec = self.model.tstats[gcontrast, fcontrast, :, :].T
        else:
            raise ValueError("Metric '{}' not recognised".format(metric))

        if title is None:
            gtitle = 'gC {} : {}'.format(gcontrast, self.contrast_names[gcontrast])
            ftitle = 'flC {} : {}'.format(fcontrast, self.fl_contrast_names[fcontrast])

            title = gtitle + '\n' + ftitle

        plot_joint_spectrum(self.f, spec, self.info, freqs=freqs, base=base,
                            topo_scale=topo_scale, lw=lw, ylabel=ylabel, title=title,
                            ylim=ylim, xtick_skip=xtick_skip, topo_prop=topo_prop, ax=ax)

    def get_fl_contrast(self, fl_con):
        """Get the data from a single first level contrast.

        Parameters
        ----------
        fl_con : int
            First level contrast data index to return

        Returns
        -------
        GroupSensorGLMSpectrum instance containing a single first level contrast.

        """
        ret_con = deepcopy(self.data)
        ret_con.data = ret_con.data[:, fl_con, :, :]

        return ret_con


class ClusterPermuteGLMSpectrum(SensorClusterPerm):
    """A class holding the result for sensor x frequency cluster stats computed
    from a group level GLM-Spectrum"""

    def plot_sig_clusters(self, thresh, ax=None, base=1):
        """Plot the significant clusters at a given threshold.

        Parameters
        ----------
        thresh : float
            The threshold to consider a cluster significant eg 95 or 99
        ax :
             (Default value = None)
        base :
             (Default value = 1)

        Returns
        -------

        """
        title = 'group-con: {}\nfirst-level-con: {}'
        title = title.format(self.gl_contrast_name, self.fl_contrast_name)

        clu, obs = self.perms.get_sig_clusters(thresh, self.perm_data)
        plot_joint_spectrum_clusters(self.f, obs, clu, self.info, base=base, ax=ax, title=title)


#%% ---------------------------------------
#
# GLM-Spectrum helper functions that take raw objects (or numpy arrays with MNE
# dim-order), compute GLM-Spectra and return class objects containing the
# result

def group_glm_spectrum(inspectra, design_config=None, datainfo=None, metric='copes'):
    """Compute a group GLM-Spectrum from a set of first-level GLM-Spectra.

    Parameters
    ----------
    inspectra : list, tuple
        A list containing either the filepaths of a set of saved GLM-Spectra
        objects, or the GLM-Spectra objects themselves.
    design_config : glmtools.design.DesignConfig
         The design specification for the group level model (Default value = None)
    datainfo : dict
         Dictionary of data values to use as covariates. The length of each
         covariate must match the number of input GLM-Spectra (Default value =
         None)
    metric : {'copes', or 'tsats'}
         Which metric to plot (Default value = 'copes')

    Returns
    -------
    GroupSensorGLMSpectrum

    References
    ----------
    .. [1] Quinn, A. J., Atkinson, L., Gohil, C., Kohl, O., Pitt, J., Zich, C., Nobre,
       A. C., & Woolrich, M. W. (2022). The GLM-Spectrum: A multilevel framework
       for spectrum analysis with covariate and confound modelling. Cold Spring
       Harbor Laboratory. https://doi.org/10.1101/2022.11.14.516449


    """
    datainfo = {} if datainfo is None else datainfo

    fl_data = []
    ## Need to sanity check that info and configs match before concatenating
    for ii in range(len(inspectra)):
        if isinstance(inspectra[ii], str):
            glmsp = read_glm_spectrum(inspectra[ii])
        else:
            glmsp = inspectra[ii]

        fl_data.append(getattr(glmsp.model, metric)[np.newaxis, ...])
        fl_contrast_names = glmsp.design.contrast_names

    fl_data = np.concatenate(fl_data, axis=0)
    group_data = glm.data.TrialGLMData(data=fl_data, **datainfo)

    if design_config is None:
        design_config = glm.design.DesignConfig()
        design_config.add_regressor(name='Mean', rtype='Constant')
        design_config.add_simple_contrasts()

    design = design_config.design_from_datainfo(group_data.info)
    model = glm.fit.OLSModel(design, group_data)

    return GroupSensorGLMSpectrum(model, design, glmsp.config, glmsp.info, data=group_data, fl_contrast_names=fl_contrast_names)


def glm_spectrum(XX, reg_categorical=None, reg_ztrans=None, reg_unitmax=None,
                 contrasts=None, fit_intercept=True, standardise_data=False,
                 window_type='hann', nperseg=None, noverlap=None, nfft=None,
                 detrend='constant', return_onesided=True, scaling='density',
                 mode='psd', fmin=None, fmax=None, axis=-1, fs=1):
    """Compute a GLM-Spectrum from a MNE-Python Raw data object.

    Parameters
    ----------
    XX : {MNE Raw object, or data array}
        Data to compute GLM-Spectrum from
    reg_categorical : dict or None
        Dictionary of covariate time series to be added as binary regessors. (Default value = None)
    reg_ztrans : dict or None
        Dictionary of covariate time series to be added as z-standardised regessors. (Default value = None)
    reg_unitmax : dict or None
        Dictionary of confound time series to be added as positive-valued unitmax regessors. (Default value = None)
    contrasts :
         (Default value = None)
    fit_intercept : bool
        Specifies whether a constant valued 'intercept' regressor is included in the model. (Default value = True)
    standardise_data : bool
        Flag indicating whether to z-transform input data (Default value = False)
    window_type :
         (Default value = 'hann')
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density')
        where `Pxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'
    mode : 
         (Default value = 'psd')
    fmin : float or None, optional
        Minimum frequency value to return (Default value = 0)
    fmax : float or None, optional
        Maximum frequency value to return (Default value = 0.5)
    axis : int
         Axis to compute spectrum over, overridden if input is an MNE raw
         object (Default value = -1)
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0. Overridden
        by value in XX.info['sfreq'] if input is a MNE Raw object.

    Returns
    -------
    SensorGLMSpectrum

    References
    ----------
    .. [1] Quinn, A. J., Atkinson, L., Gohil, C., Kohl, O., Pitt, J., Zich, C., Nobre,
       A. C., & Woolrich, M. W. (2022). The GLM-Spectrum: A multilevel framework
       for spectrum analysis with covariate and confound modelling. Cold Spring
       Harbor Laboratory. https://doi.org/10.1101/2022.11.14.516449

    """
    if isinstance(XX, mne.io.base.BaseRaw):
        fs = XX.info['sfreq']
        nperseg = int(np.floor(fs)) if nperseg is None else nperseg
        YY = XX.get_data()
        axis = 1
    else:
        YY = XX

    if standardise_data:
        YY = stats.zscore(YY, axis=axis)

    # sails.sftf.config freqvals isn't right when frange is trimmed!
    glmsp = glm_periodogram(YY, axis=axis,
                            reg_categorical=reg_categorical,
                            reg_ztrans=reg_ztrans,
                            reg_unitmax=reg_unitmax,
                            contrasts=contrasts,
                            fit_intercept=fit_intercept,
                            window_type=window_type,
                            fs=fs,
                            nperseg=nperseg,
                            noverlap=noverlap,
                            nfft=nfft,
                            detrend=detrend,
                            return_onesided=return_onesided,
                            scaling=scaling,
                            mode=mode,
                            fmin=fmin,
                            fmax=fmax,
                            ret_class=True,
                            fit_method='glmtools')

    if isinstance(XX, mne.io.base.BaseRaw):
        return SensorGLMSpectrum(glmsp, XX.info)
    else:
        return glmsp


def read_glm_spectrum(infile):
    """Read in a GLMSpectrum object that has been saved as as a pickle.

    Parameters
    ----------
    infile : str
        Filepath of saved object

    Returns
    -------
    SensorGLMSpectrum

    """
    with open(infile, 'rb') as outp:
        glmsp = pickle.load(outp)
    return glmsp

##% ---------------------------
#
# Plotting!


def plot_joint_spectrum_clusters(xvect, psd, clusters, info, ax=None, freqs='auto', base=1,
                                 topo_scale='joint', lw=0.5, ylabel='Power', title='', ylim=None,
                                 xtick_skip=1, topo_prop=1/5):
    """

    Parameters
    ----------
    xvect :

    psd :

    clusters :

    info :

    ax :
         (Default value = None)
    freqs :
         (Default value = 'auto')
    base :
         (Default value = 1)
    topo_scale :
         (Default value = 'joint')
    lw :
         (Default value = 0.5)
    ylabel :
         (Default value = 'Power')
    title :
         (Default value = '')
    ylim :
         (Default value = None)
    xtick_skip :
         (Default value = 1)
    topo_prop :
         (Default value = 1/3)

    Returns
    -------

    """
    if ax is None:
        fig = plt.figure()
        fig.subplots_adjust(top=0.8)
        ax = plt.subplot(111)

    ax.set_axis_off()

    title_prop = 0.1
    main_prop = 1-title_prop-topo_prop
    main_ax = ax.inset_axes((0, 0, 1, main_prop))

    plot_sensor_spectrum(xvect, psd, info, ax=main_ax, base=base, lw=0.25, ylabel=ylabel)
    fx = prep_scaled_freq(base, xvect)

    yl = main_ax.get_ylim()
    main_ax.set_ylim(yl[0], 1.2*yl[1])

    yt = ax.get_yticks()
    inds = yt < yl[1]
    ax.set_yticks(yt[inds])

    ax.figure.canvas.draw()
    offset = ax.yaxis.get_major_formatter().get_offset()
    ax.yaxis.offsetText.set_visible(False)
    ax.text(0, yl[1], offset, ha='right')

    if len(clusters) == 0:
        # put up an empty axes anyway
        topo_pos = [0.3, 1.2, 0.4, 0.4]
        topo = ax.inset_axes(topo_pos, frame_on=False)
        topo.set_xticks([])
        topo.set_yticks([])

    # Reorder clusters in ascending frequency
    clu_order = []
    for clu in clusters:
        clu_order.append(clu[2][0].min())
    clusters = [clusters[ii] for ii in np.argsort(clu_order)]

    print('\n')
    table_header = '{0:12s}{1:16s}{2:12s}{3:12s}{4:14s}'
    print(table_header.format('Cluster', 'Statistic', 'Freq Min', 'Freq Max', 'Num Channels'))
    table_template = '{0:<12d}{1:<16.3f}{2:<12.2f}{3:<12.2f}{4:<14d}'

    topo_centres = np.linspace(0, 1, len(clusters)+2)[1:-1]
    topo_width = 0.4
    topos = []
    ymax_span = (np.abs(yl[0]) + yl[1]) / (np.abs(yl[0]) + yl[1]*1.2)
    for idx in range(len(clusters)):
        clu = clusters[idx]

        # Create topomap axis
        topo_pos = [topo_centres[idx] - 0.2, 1-title_prop-topo_prop, 0.4, topo_prop]
        topo_ax = ax.inset_axes(topo_pos)

        # Extract cluster location in space and frequency
        channels = np.zeros((psd.shape[1], ))
        channels[clu[2][1]] = 1
        if len(channels) == 204:
            channels = np.logical_or(channels[::2], channels[1::2])
        freqs = np.zeros((psd.shape[0], ))
        freqs[clu[2][0]] = 1
        finds = np.where(freqs)[0]
        if len(finds) == 1:
            finds = np.array([finds[0], finds[0]+1])

        msg = 'Cluster {} - stat: {}, freq range: {}, num channels {}'
        freq_range = (fx[0][finds[0]], fx[0][finds[-1]])
        print(table_template.format(idx+1, clu[0], freq_range[0], freq_range[1], int(channels.sum())))

        # Plot cluster span overlay on spectrum
        main_ax.axvspan(fx[0][finds[0]], fx[0][finds[-1]], facecolor=[0.7, 0.7, 0.7], alpha=0.5, ymax=ymax_span)
        fmid = int(np.floor(finds.mean()))

        # Plot connecting line to topo
        xy_main = (fx[0][fmid], yl[1])
        xy_topo = (0.5, 0)
        con = ConnectionPatch(
            xyA=xy_main, coordsA=main_ax.transData,
            xyB=xy_topo, coordsB=topo_ax.transAxes,
            arrowstyle="-", color=[0.7, 0.7, 0.7])
        main_ax.figure.add_artist(con)

        # Plot topo
        dat = psd[fmid, :]
        im, cn = mne.viz.plot_topomap(dat, info, axes=topo_ax, show=False, mask=channels)
        topos.append(im)

    if topo_scale == 'joint':
        vmin = np.min([t.get_clim()[0] for t in topos])
        vmax = np.max([t.get_clim()[1] for t in topos])

        for t in topos:
            t.set_clim(vmin, vmax)
    else:
        vmin = 0
        vmax = 1
    print('\n')  # End table

    cb_pos = [0.95, 1-title_prop-topo_prop, 0.025, topo_prop]
    cax =  ax.inset_axes(cb_pos)

    plt.colorbar(topos[0], cax=cax)

    ax.set_title(title, x=0.5, y=1-title_prop)


def plot_joint_spectrum(xvect, psd, info, ax=None, freqs='auto', base=1,
        topo_scale='joint', lw=0.5, ylabel='Power', title='', ylim=None,
        xtick_skip=1, topo_prop=1/5):
    """Plot a GLM-Spectrum contrast with spatial line colouring and topograpies.

    Parameters
    ----------
    xvect : array_like
        Vector of frequency values for x-axis
    psd : array_like
        Array of spectrum values to plot
    info : MNE Raw info
        Sensor info for spatial map
    ax : {None or axis handle}
         Axis to plot into (Default value = None)
    freqs : {list, tuple or 'auto'}
         Which frequencies to plot topos for (Default value = 'auto')
    base : float
         The x-axis scaling, set to 0.5 for sqrt freq axis (Default value = 1)
    topo_scale : {'joint' or None}
         Whether to fix topomap colour scales across all topos ('joint') or
         leave them individual (Default value = 'joint')
    lw : flot
         Line width(Default value = 0.5)
    ylabel : str
         Y-axis label(Default value = 'Power')
    title : str
         Plot title(Default value = None)
    ylim : {tuple or list}
         min and max values for y-axis (Default value = None)
    xtick_skip : int
         Number of xaxis ticks to skip, useful for tight plots (Default value = 1)
    topo_prop : float
         Proportion of plot dedicted to topomaps(Default value = 1/3)

    """
    if ax is None:
        fig = plt.figure()
        fig.subplots_adjust(top=0.8)
        ax = plt.subplot(111)

    ax.set_axis_off()

    title_prop = 0.1
    main_prop = 1-title_prop-topo_prop
    main_ax = ax.inset_axes((0, 0, 1, main_prop))

    plot_sensor_spectrum(xvect, psd, info, ax=main_ax, base=base, lw=0.25, ylabel=ylabel)
    fx = prep_scaled_freq(base, xvect)

    if freqs == 'auto':
        topo_freq_inds = signal.find_peaks(psd.mean(axis=1), distance=xvect.shape[0]/3)[0]
        if len(topo_freq_inds) > 2:
            I = np.argsort(psd.mean(axis=1)[topo_freq_inds])[-2:]
            topo_freq_inds = topo_freq_inds[I]
        freqs = xvect[topo_freq_inds]
    else:
        topo_freq_inds = [np.argmin(np.abs(xvect - ff)) for ff in freqs]

    yl = main_ax.get_ylim()
    main_ax.set_ylim(yl[0], 1.2*yl[1])

    yt = ax.get_yticks()
    inds = yt < yl[1]
    ax.set_yticks(yt[inds])

    ax.figure.canvas.draw()
    offset = ax.yaxis.get_major_formatter().get_offset()
    ax.yaxis.offsetText.set_visible(False)
    ax.text(0, yl[1], offset, ha='right')

    topo_centres = np.linspace(0, 1, len(freqs)+2)[1:-1]
    topo_width = 0.4
    topos = []
    for idx in range(len(freqs)):
        # Create topomap axis
        topo_pos = [topo_centres[idx] - 0.2, 1-title_prop-topo_prop, 0.4, topo_prop]
        topo_ax = ax.inset_axes(topo_pos)

        topo_idx = fx[0][topo_freq_inds[idx]]
        main_ax.plot((topo_idx, topo_idx), yl, color=[0.7, 0.7, 0.7])

        xy_main = (topo_idx, yl[1])
        xy_topo = (0.5, 0)
        con = ConnectionPatch(
            xyA=xy_main, coordsA=main_ax.transData,
            xyB=xy_topo, coordsB=topo_ax.transAxes,
            arrowstyle="-", color=[0.7, 0.7, 0.7])
        ax.figure.add_artist(con)

        dat = psd[topo_freq_inds[idx], :]
        im, cn = mne.viz.plot_topomap(dat, info, axes=topo_ax, show=False)
        topos.append(im)

    if topo_scale == 'joint':
        vmin = np.min([t.get_clim()[0] for t in topos])
        vmax = np.max([t.get_clim()[1] for t in topos])

        for t in topos:
            t.set_clim(vmin, vmax)
    else:
        vmin = 0
        vmax = 1

    cb_pos = [0.95, 1-title_prop-topo_prop, 0.025, topo_prop]
    cax =  ax.inset_axes(cb_pos)

    plt.colorbar(topos[0], cax=cax)

    ax.set_title(title, x=0.5, y=1-title_prop)


def plot_sensor_spectrum(xvect, psd, info, ax=None, sensor_proj=False,
                         xticks=None, xticklabels=None, lw=0.5, title=None,
                         sensor_cols=True, base=1, ylabel=None, xtick_skip=1):
    """

    Parameters
    ----------
    xvect :

    psd :

    info :

    ax :
         (Default value = None)
    sensor_proj :
         (Default value = False)
    xticks :
         (Default value = None)
    xticklabels :
         (Default value = None)
    lw :
         (Default value = 0.5)
    title :
         (Default value = None)
    sensor_cols :
         (Default value = True)
    base :
         (Default value = 1)
    ylabel :
         (Default value = None)
    xtick_skip :
         (Default value = 1)

    Returns
    -------

    """

    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    plot_sensor_data(xvect, psd, info, ax=ax, base=base,
                     sensor_cols=sensor_cols, lw=lw, xticks=xticks,
                     xticklabels=xticklabels, xtick_skip=xtick_skip)
    decorate_spectrum(ax, ylabel=ylabel)
    ax.set_ylim(psd.min())

    if sensor_proj:
        axins = ax.inset_axes([0.6, 0.6, 0.37, 0.37])
        plot_channel_layout(axins, info)

    if title is not None:
        ax.set_title(title)


def plot_sensor_data(xvect, data, info, ax=None, lw=0.5,
                     xticks=None, xticklabels=None,
                     sensor_cols=True, base=1, xtick_skip=1):
    """

    Parameters
    ----------
    xvect :

    data :

    info :

    ax :
         (Default value = None)
    lw :
         (Default value = 0.5)
    xticks :
         (Default value = None)
    xticklabels :
         (Default value = None)
    sensor_cols :
         (Default value = True)
    base :
         (Default value = 1)
    xtick_skip :
         (Default value = 1)

    Returns
    -------

    """

    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    fx, xticklabels, xticks = prep_scaled_freq(base, xvect)

    if sensor_cols:
        colors, pos, outlines = get_mne_sensor_cols(info)
    else:
        colors = None

    plot_with_cols(ax, data, fx, colors, lw=lw)
    ax.set_xlim(fx[0], fx[-1])

    if xticks is not None:
        ax.set_xticks(xticks[::xtick_skip])
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels[::xtick_skip])


def prep_scaled_freq(base, freq_vect):
    """Assuming ephy freq ranges for now - around 1-40Hz

    Parameters
    ----------
    base :

    freq_vect :


    Returns
    -------

    """
    fx = freq_vect**base
    if base < 1:
        nticks = int(np.floor(np.sqrt(freq_vect[-1])))
        ftick = np.array([ii**2 for ii in range(1,nticks+1)])
        ftickscaled = ftick**base
    else:
        # Stick with automatic scales
        ftick = None
        ftickscaled = None
    return fx, ftick, ftickscaled


def get_mne_sensor_cols(info):
    """

    Parameters
    ----------
    info :


    Returns
    -------

    """

    chs = [info['chs'][i] for i in range(len(info['chs']))]
    locs3d = np.array([ch['loc'][:3] for ch in chs])
    x, y, z = locs3d.T
    colors = mne.viz.evoked._rgb(x, y, z)
    pos, outlines = mne.viz.evoked._get_pos_outlines(info,
                                                     range(len(info['chs'])),
                                                     sphere=None)

    return colors, pos, outlines


def plot_channel_layout(ax, info, size=30, marker='o'):
    """

    Parameters
    ----------
    ax :

    info :

    size :
         (Default value = 30)
    marker :
         (Default value = 'o')

    Returns
    -------

    """

    ax.set_adjustable('box')
    ax.set_aspect('equal')

    colors, pos, outlines = get_mne_sensor_cols(info)
    pos_x, pos_y = pos.T
    mne.viz.evoked._prepare_topomap(pos, ax, check_nonzero=False)
    ax.scatter(pos_x, pos_y,
               color=colors, s=size * .8,
               marker=marker, zorder=1)
    mne.viz.evoked._draw_outlines(ax, outlines)


def plot_with_cols(ax, data, xvect, cols=None, lw=0.5):
    """

    Parameters
    ----------
    ax :

    data :

    xvect :

    cols :
         (Default value = None)
    lw :
         (Default value = 0.5)

    Returns
    -------

    """
    if cols is not None:
        for ii in range(data.shape[1]):
            ax.plot(xvect, data[:, ii], lw=lw, color=cols[ii, :])
    else:
        ax.plot(xvect, data, lw=lw)


def decorate_spectrum(ax, ylabel='Power'):
    """

    Parameters
    ----------
    ax :

    ylabel :
         (Default value = 'Power')

    Returns
    -------

    """
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(ylabel)
