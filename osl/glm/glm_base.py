
import pickle
from pathlib import Path

import mne
import numpy as np
from scipy.sparse import csr_array

import glmtools as glm

from ..source_recon import parcellation


class GLMBaseResult:
    """A class for GLM-Epochs fitted to MNE-Python Raw objects."""
    def __init__(self, model, design, info, data=None):
        """
        Parameters
        ----------
        model : :py:class:`glmtools.fit.OLSModel <glmtools.fit.OLSModel>'
            The  model object.
        design : :py:class:`glmtools.design.GLMDesign <glmtools.design.GLMDesign>`
            The  GLM design object.
        info : :py:class:`mne.Info <mne.Info>`
            The MNE-Python Info object.
            
            
        References
        ----------
        https://gitlab.com/ajquinn/glmtools/-/tree/master
        
        """
        self.model = model
        self.design = design
        self.data = data
        self.info = info


    def save_pkl(self, outname, overwrite=True, save_data=False):
        """Save GLM-Epochs result to a pickle file.

        Parameters
        ----------
        outname : str
             Filename or full file path to write pickle to
        overwrite : bool
             Overwrite previous file if one exists? (Default value = True)
        save_data : bool
             Save epoch data in pickle? This is omitted by default to save disk
             space (Default value = False)

        """
        if Path(outname).exists() and not overwrite:
            msg = "{} already exists. Please delete or do use overwrite=True."
            raise ValueError(msg.format(outname))

        if hasattr(self, 'config'):
            self.config.detrend_func = None  # Have to drop this to pickle glm-spectra

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


class GroupGLMBaseResult:
    """A class for group level GLM-Epochs fitted across mmultiple first-level
    GLM-Epochs computed from MNE-Python Raw objects"""

    def __init__(self, model, design, info, config, fl_contrast_names=None, data=None):
        """
        Parameters
        ----------
        model : :py:class:`glmtools.fit.OLSModel <glmtools.fit.OLSModel>'
            The  model object.
        design : :py:class:`glmtools.design.GLMDesign <glmtools.design.GLMDesign>`
            The  GLM design object.
        info : :py:class:`mne.Info <mne.Info>`
            The MNE-Python Info object.
        config : :py:class:`glmtools.config.GLMConfig <glmtools.config.GLMConfig>`
            The  GLM configuration object.
        fl_contrast_names : list of str
            The names of the first-level contrasts.
        data : :py:class:`mne.Epochs <mne.Epochs>` or :py:class:`mne.Evoked <mne.Evoked>`
            The MNE-Python Epochs or Evoked object.
            
        """
        
        self.model = model
        self.design = design
        self.data = data
        self.config = config

        self.info = info

        # A proper group-model in glmtools will simplify this
        self.contrast_names = self.model.contrast_names
        if fl_contrast_names is None:
            self.fl_contrast_names = [chr(65 + ii) for ii in range(self.model.copes.shape[1])]
        else:
            self.fl_contrast_names = fl_contrast_names

    def get_channel_adjacency(self, dist=40):
        """Return adjacency matrix of channels.
        
        Parameters
        ----------
        dist : float
            Distance in mm between parcel centroids to consider neighbours.
            Only used if data is parcellated.
        
        Returns
        -------
        adjacency : scipy.sparse.csr_matrix
            The adjacency matrix.
        ch_names : list of str
            The channel names.
        """
        if np.any(['parcel' in ch for ch in self.info['ch_names']]):
            # We have parcellated data
            parcellation_file = parcellation.guess_parcellation(int(np.sum(['parcel' in ch for ch in self.info['ch_names']])))
            adjacency = csr_array(parcellation.spatial_dist_adjacency(parcellation_file, dist=dist))
        elif np.any(['state' in ch for ch in self.info['ch_names']]) or np.any(['mode' in ch for ch in self.info['ch_names']]):
            adjacency = csr_array(np.eye(len(self.info['ch_names'])))
        else:
            ch_type =  mne.io.meas_info._get_channel_types(self.info)[0]  # Assuming these are all the same!
            adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(self.info, ch_type)
        ntests = np.prod(self.data.data.shape[2:])
        ntimes = self.data.data.shape[3]
        print('{} : {}'.format(ntimes, ntests))
        return mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)


class BaseSensorPerm:
    """A base class for sensor x frequency permutation tests computed from a
    group level GLM-Spectrum."""
    def save_pkl(self, outname, overwrite=True, save_data=False):
        """Save GLM-Epochs result to a pickle file.

        Parameters
        ----------
        outname : str
                Filename or full file path to write pickle to
        overwrite : bool
                Overwrite previous file if one exists? (Default value = True)
        save_data : bool
                Save epoch data in pickle? This is omitted by default to save disk
                space (Default value = False)

        """
        if Path(outname).exists() and not overwrite:
            msg = "{} already exists. Please delete or do use overwrite=True."
            raise ValueError(msg.format(outname))

        with open(outname, 'bw') as outp:
            pickle.dump(self, outp)


class SensorMaxStatPerm(BaseSensorPerm):
    """A class holding the result for sensor x frequency max-stat permutation test computed
    from a group level GLM-Spectrum"""

    def __init__(self, glmsp, gl_con, fl_con=0, nperms=1000,
                    tstat_args=None,  metric='tstats', nprocesses=1,
                    pooled_dims=(1,2)):
        """Initialise the SensorMaxStatPerm class.
        
        Parameters
        ----------
        glmsp : :py:class:`GroupGLMBaseResult <GroupGLMBaseResult>`
            The group level GLM result object.
        gl_con : int
            The index of the group level contrast to test.
        fl_con : int
            The index of the first-level contrast to test.
        nperms : int
            The number of permutations to use.
        tstat_args : dict
            The arguments to pass to the tstat function.
        metric : str
            The metric to use for the permutation test.
        nprocesses : int
            The number of processes to use.
        pooled_dims : tuple of int
            The dimensions to pool over.
        
        """
        # There is a major pain here in that MNE stores raw data in [channels x time]
        # but builds adjacencies in [time x channels] - we don't need adjacencies for perms
        # but we do for making clusters for plotting, so here we are
        self.adjacency = glmsp.get_channel_adjacency()
        self.perm_data = glmsp.get_fl_contrast(fl_con)
        self.perm_data.data = np.swapaxes(self.perm_data.data, 1, 2)

        # change the dim_labels to match the data (swap time and channels, and remove firstlevel contrast)
        self.perm_data.info['dim_labels'] = [self.perm_data.info['dim_labels'][i] for i in [0,3,2]]
                        
        self.gl_con = gl_con
        self.fl_con = fl_con
        self.gl_contrast_name = glmsp.contrast_names[gl_con]
        self.fl_contrast_name = glmsp.fl_contrast_names[fl_con]
        self.info = glmsp.info
        self.f = glmsp.f

        self.perms = glm.permutations.MaxStatPermutation(glmsp.design, self.perm_data, gl_con, nperms,
                                                        nprocesses=nprocesses,
                                                        metric=metric,
                                                        pooled_dims=pooled_dims,
                                                        tstat_args=tstat_args)

    def get_sig_clusters(self, thresh):
        """Return the significant clusters at a given threshold.
        
        Parameters
        ----------
        thresh : float
            The threshold to consider a cluster significant eg 95 or 99
            
        Returns
        -------
        clusters
            A list containing the significant clusters. Each list item contains
            a tuple of three items - the cluster statistic, the cluster
            percentile relative to the null and the spatial/spectral indices of
            the cluster.
        obs_stat
            The observed statistic.
        """
        obs = glm.fit.OLSModel(self.perms._design, self.perm_data)
        obs = obs.get_tstats(**self.perms.tstat_args)[self.gl_con, :, :]
        thresh = self.perms.get_thresh(thresh)

        obs_up = obs.flatten() > thresh
        obs_down = obs.flatten() < -thresh

        from mne.stats.cluster_level import _find_clusters as mne_find_clusters
        from mne.stats.cluster_level import _reshape_clusters as mne_reshape_clusters

        clus_up, cstat_up = mne_find_clusters(obs_up, 0.5, adjacency=self.adjacency)
        clus_up = mne_reshape_clusters(clus_up, obs.shape)

        clus_down, cstat_down = mne_find_clusters(obs_down, 0.5, adjacency=self.adjacency)
        clus_down = mne_reshape_clusters(clus_down, obs.shape)

        # cstat, pval, clu - match cluster stat output
        clusters = []
        for ii in range(len(cstat_down)):
            clusters.append([cstat_down[ii], 0, clus_down[ii]])
        for ii in range(len(cstat_up)):
            clusters.append([cstat_up[ii], 0, clus_up[ii]])

        return clusters, obs


class SensorClusterPerm(BaseSensorPerm):
    """A class holding the result for sensor x frequency cluster stats computed
    from a group level GLM-Spectrum"""

    def __init__(self, glmsp, gl_con, fl_con=0, nperms=1000,
                    cluster_forming_threshold=3, tstat_args=None,
                    metric='tstats', nprocesses=1):
        """Initialise the SensorClusterPerm class.
        
        Parameters
        ----------
        glmsp : :py:class:`GroupGLMBaseResult <GroupGLMBaseResult>`
            The group level GLM result object.
        gl_con : int
            The index of the group level contrast to test.
        fl_con : int
            The index of the first-level contrast to test.
        nperms : int
            The number of permutations to use.
        cluster_forming_threshold : float   
            The threshold to use for cluster forming.
        tstat_args : dict   
            The arguments to pass to the tstat function.
        metric : str
            The metric to use for the permutation test.
        nprocesses : int    
            The number of processes to use.
        
        """

        # There is a major pain here in that MNE stores raw data in [channels x time]
        # but builds adjacencies in [time x channels]
        self.perm_data = glmsp.get_fl_contrast(fl_con)
        self.perm_data.data = np.swapaxes(self.perm_data.data, 1, 2)

        self.gl_contrast_name = glmsp.contrast_names[gl_con]
        self.fl_contrast_name = glmsp.fl_contrast_names[fl_con]
        self.info = glmsp.info
        self.f = glmsp.f

        self.perms = glm.permutations.MNEClusterPermutation(glmsp.design, self.perm_data, gl_con, nperms,
                                                        nprocesses=nprocesses,
                                                        metric=metric,
                                                        cluster_forming_threshold=cluster_forming_threshold,
                                                        tstat_args=tstat_args,
                                                        adjacency=glmsp.get_channel_adjacency())

    def get_sig_clusters(self, thresh):
        """Return the significant clusters at a given threshold.

        Parameters
        ----------
        thresh : float
            The threshold to consider a cluster significant eg 95 or 99

        Returns
        -------
        clusters
            A list containing the significant clusters. Each list item contains
            a tuple of three items - the cluster statistic, the cluster
            percentile relative to the null and the spatial/spectral indices of
            the cluster.
        obs_stat
            The observed statistic.
        """
        clusters, obs_stat =  self.perms.get_sig_clusters(thresh, self.perm_data)
        return clusters, obs_stat
