
import numpy as np
import mne
import glmtools as glm
import matplotlib.pyplot as plt
from scipy import signal, stats


class GLMEpochsResult:

    def __init__(self, model, design, info, tmin=0):

        self.model = model
        self.design = design
        self.info = info
        self.tmin = tmin

    def get_evoked_contrast(self, contrast=0, metric='copes'):

        if metric == 'copes':
            erf = self.model.copes[contrast, :, :]
        elif metric == 'tstats':
            erf = self.model.tstats[contrast, :, :]

        return mne.EvokedArray(erf, self.info, tmin=self.tmin)

    def plot_joint_contrast(self, contrast=0, metric='copes', title=None):

        evo = self.get_evoked_contrast(contrast=contrast, metric=metric)

        if title is None:
            title = 'C {} : {}'.format(contrast, self.design.contrast_names[contrast])

        evo.plot_joint(title=title)


def glm_epochs(config, epochs):

    data = load_mne_epochs(epochs)
    design = config.design_from_datainfo(data.info)

    model = glm.fit.OLSModel(design, data)

    return GLMEpochsResult(model, design, epochs.info, tmin=epochs.tmin)


def load_mne_epochs(X):

    import mne
    if isinstance(X, str) and os.path.isfile(X):
        epochs = mne.read_epochs(X)
    elif isinstance(X, (mne.epochs.EpochsFIF, mne.Epochs)):
        epochs = X

    d = glm.data.TrialGLMData(data=epochs.get_data(),
                              category_list=epochs.events[:, 2],
                              sample_rate=epochs.info['sfreq'],
                              time_dim=2,
                              dim_labels=list(('Trials', 'Channels', 'Times')))

    return d

