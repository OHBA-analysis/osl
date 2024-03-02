
import os
import pickle
from copy import deepcopy

import glmtools as glm
import mne
import numpy as np

from .glm_base import GLMBaseResult, GroupGLMBaseResult, SensorClusterPerm


class GLMEpochsResult(GLMBaseResult):
    """A class for first-level GLM-Spectra fitted to MNE-Python Epochs objects"""
    def __init__(self, model, design, info, tmin=0, data=None, times=None):
        """
        Parameters
        ----------
        model : :py:class:`glmtools.fit.OLSModel <glmtools.fit.OLSModel>'
            The  model object.
        design : :py:class:`glmtools.design.GLMDesign <glmtools.design.GLMDesign>`
            The  GLM design object.
        info : mne.Info
            The MNE-Python Info object for the data
        tmin : float
            The time of the first time point in the epoch (Default value = 0)
        data : glmtools.data.TrialGLMData
            The data object used to fit the model (Default value = None)
        times : array-like
            The time points for the data (Default value = None)
        
        """
        self.tmin = tmin
        self.times = times
        super().__init__(model, design, info, data=data)

    def get_evoked_contrast(self, contrast=0, metric='copes'):
        """Get the evoked response for a given contrast.
        
        Parameters
        ----------
        contrast : int
            Contrast index to return
        metric : {'copes', or 'tsats'}
            Which metric to plot (Default value = 'copes')
            
        Returns
        -------
        :py:class:`mne.Evoked <mne.Evoked>`
            The evoked response for the contrast.
        """
        if metric == 'copes':
            erf = self.model.copes[contrast, :, :]
        elif metric == 'tstats':
            erf = self.model.tstats[contrast, :, :]

        return mne.EvokedArray(erf, self.info, tmin=self.tmin)

    def plot_joint_contrast(self, contrast=0, metric='copes', title=None):
        """Plot the evoked response for a given contrast.
        
        Parameters
        ----------
        contrast : int
            Contrast index to return
        metric : {'copes', or 'tsats'}
            Which metric to plot (Default value = 'copes')
        title : str
            Title for the plot
        """
        evo = self.get_evoked_contrast(contrast=contrast, metric=metric)

        if title is None:
            title = 'C {} : {}'.format(contrast, self.design.contrast_names[contrast])

        evo.plot_joint(title=title)


class GroupGLMEpochs(GroupGLMBaseResult):
    """A class for group level GLM-Spectra fitted across mmultiple first-level
    GLM-Spectra computed from MNE-Python Raw objects"""

    def __init__(self, model, design, info, fl_contrast_names=None, data=None, tmin=0, times=None):
        """
        Parameters
        ----------
        model : :py:class:`glmtools.fit.OLSModel <glmtools.fit.OLSModel>'
            The  model object.
        design : :py:class:`glmtools.design.GLMDesign <glmtools.design.GLMDesign>`
            The  GLM design object.
        info : mne.Info
            The MNE-Python Info object for the data
        fl_contrast_names : {None, list}
            List of first-level contrast names (Default value = None)
        data : glmtools.data.TrialGLMData
            The data object used to fit the model (Default value = None)
        tmin : float
            The time of the first time point in the epoch (Default value = 0)
        times : array-like
            The time points for the data (Default value = None)
        """
        self.tmin = tmin
        self.times = times
        super().__init__(model, design, info, fl_contrast_names=fl_contrast_names, data=data)

    def get_evoked_contrast(self, gcontrast=0, fcontrast=0, metric='copes'):
        """Get the evoked response for a given contrast.
        
        Parameters
        ----------
        contrast : int
            Contrast index to return
        metric : {'copes', or 'tsats'}
            Which metric to plot (Default value = 'copes')
            
        Returns
        -------
        :py:class:`mne.Evoked <mne.Evoked>`
            The evoked response for the contrast.
        """
        if metric == 'copes':
            erf = self.model.copes[gcontrast, fcontrast, :, :]
        elif metric == 'tstats':
            erf = self.model.tstats[gcontrast, fcontrast, :, :]

        return mne.EvokedArray(erf, self.info, tmin=self.tmin)

    def plot_joint_contrast(self, gcontrast=0, fcontrast=0, metric='copes', title=None, joint_args=None):
        """Plot the evoked response for a given contrast.
        
        Parameters
        ----------
        contrast : int
            Contrast index to return
        metric : {'copes', or 'tsats'}
            Which metric to plot (Default value = 'copes')
        title : str
            Title for the plot
        """
        evo = self.get_evoked_contrast(gcontrast=0, fcontrast=0, metric=metric)

        joint_args = {} if joint_args is None else joint_args
        if title is None:
            gtitle = 'gC {} : {}'.format(gcontrast, self.contrast_names[gcontrast])
            ftitle = 'flC {} : {}'.format(fcontrast, self.fl_contrast_names[fcontrast])

            title = gtitle + '\n' + ftitle

        if metric == 'tstats':
            joint_args['ts_args'] = {'scalings': dict(eeg=1, grad=1, mag=1),
                                     'units': dict(eeg='tstats', grad='tstats', mag='tstats')}

        evo.plot_joint(title=title, **joint_args)

    def get_channel_adjacency(self):
        """Return adjacency matrix of channels."""
        ch_type =  mne.io.meas_info._get_channel_types(self.info)[0]  # Assuming these are all the same!
        adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(self.info, ch_type)
        ntests = np.prod(self.data.data.shape[2:])
        ntimes = self.data.data.shape[3]
        print('{} : {}'.format(ntimes, ntests))
        return mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)

    def get_fl_contrast(self, fl_con):
        """Get the data from a single first level contrast.

        Parameters
        ----------
        fl_con : int
            First level contrast data index to return

        Returns
        -------
        :py:class:`GLMEpochsResult <glmtools.glm_epochs.GLMEpochsResult>`  instance containing a single first level contrast.

        """
        ret_con = deepcopy(self.data)
        ret_con.data = ret_con.data[:, fl_con, :, :]

        return ret_con


#%% ------------------------------------------------------


def glm_epochs(config, epochs):
    """Compute a GLM-Epochs from an MNE-Python Epochs object.
    
    Parameters
    ---------- 
    config : glmtools.design.DesignConfig
         The design specification for the model
    epochs : str, :py:class:`mne.Epochs <mne.Epochs>`
         The epochs object to use for the model
    
    Returns
    -------
    :py:class:`GLMEpochsResult <glmtools.glm_epochs.GLMEpochsResult>`
         """
    data = read_mne_epochs(epochs)
    design = config.design_from_datainfo(data.info)

    model = glm.fit.OLSModel(design, data)

    return GLMEpochsResult(model, design, epochs.info, tmin=epochs.tmin, data=data, times=epochs.times)

def group_glm_epochs(inspectra, design_config=None, datainfo=None, metric='copes', baseline=None):
    """Compute a group GLM-Epochs from a set of first-level GLM-Epochs.

    Parameters
    ----------
    inspectra : list, tuple
        A list containing either the filepaths of a set of saved GLM-Epochs
        objects, or the GLM-Epochs objects themselves.
    design_config : glmtools.design.DesignConfig
         The design specification for the group level model (Default value = None)
    datainfo : dict
         Dictionary of data values to use as covariates. The length of each
         covariate must match the number of input GLM-Epochs (Default value =
         None)
    metric : {'copes', or 'tsats'}
         Which metric to plot (Default value = 'copes')

    Returns
    -------
    :py:class:`GroupGLMEpochs <glmtools.glm_epochs.GroupGLMEpochs>`

    """
    datainfo = {} if datainfo is None else datainfo

    fl_data = []
    ## Need to sanity check that info and configs match before concatenating
    for ii in range(len(inspectra)):
        if isinstance(inspectra[ii], str):
            glmep = read_glm_epochs(inspectra[ii])
        else:
            glmep = inspectra[ii]

        fl_data.append(getattr(glmep.model, metric)[np.newaxis, ...])
        fl_contrast_names = glmep.design.contrast_names

    fl_data = np.concatenate(fl_data, axis=0)
    group_data = glm.data.TrialGLMData(data=fl_data, **datainfo)

    if design_config is None:
        design_config = glm.design.DesignConfig()
        design_config.add_regressor(name='Mean', rtype='Constant')
        design_config.add_simple_contrasts()

    design = design_config.design_from_datainfo(group_data.info)
    model = glm.fit.OLSModel(design, group_data)

    return GroupGLMEpochs(model, design, glmep.info, data=group_data, fl_contrast_names=fl_contrast_names, tmin=glmep.tmin, times=glmep.times)

#%% ------------------------------------------------------

def read_mne_epochs(X, picks=None):
    """Read in an MNE-Python Epochs object and convert it to a GLM data object.
    
    Parameters
    ----------
    X : str, :py:class:`mne.Epochs <mne.Epochs>`
        The epochs object to use for the model
    picks : list
        List of channels to use for the model (Default value = None)
        
    Returns
    -------
    :py:class:`glmtools.data.TrialGLMData <glmtools.data.TrialGLMData>`
        The data object used to fit the model.
    """
    import mne
    if isinstance(X, str) and os.path.isfile(X):
        epochs = mne.read_epochs(X)
    elif isinstance(X, (mne.epochs.EpochsFIF, mne.Epochs)):
        epochs = X

    d = glm.data.TrialGLMData(data=epochs.get_data(picks=picks),
                              category_list=epochs.events[:, 2],
                              sample_rate=epochs.info['sfreq'],
                              time_dim=2,
                              dim_labels=list(('Trials', 'Channels', 'Times')))

    return d

def read_glm_epochs(infile):
    """Read in a GLMEpochs object that has been saved as as a pickle.

    Parameters
    ----------
    infile : str
        Filepath of saved object

    Returns
    -------
    :py:class:`GLMEpochsResult <glmtools.glm_epochs.GLMEpochsResult>`

    """
    with open(infile, 'rb') as outp:
        glmepec = pickle.load(outp)
    return glmepec
