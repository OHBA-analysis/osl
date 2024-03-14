"""Utility functions for simulating data.

"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>

import os
import mne
import sails
import numpy as np


def simulate_data(model, num_samples=1000, num_realisations=1, use_cov=True, noise=None):
    """Simulate data from a linear model.
    
    Parameters
    ----------
    model : sails.AbstractLinearModel
        A linear model object.
    num_samples : int
        The number of samples to simulate.
    num_realisations : int
        The number of realisations to simulate.
    use_cov : bool
        Whether to use the residual covariance matrix.
        
    Returns
    -------
    Y : ndarray, shape (num_sources, num_samples, num_realisations)
        The simulated data.
    
    """
    
    num_sources = model.nsignals

    # Preallocate output
    Y = np.zeros((num_sources, num_samples, num_realisations))

    for ep in range(num_realisations):

        # Create driving noise signal
        Y[:, :, ep] = np.random.randn(num_sources, num_samples)

        if use_cov:
            C = np.linalg.cholesky(model.resid_cov)
            Y[:, :, ep] = Y[:, :, ep].T.dot(C).T

        # Main Loop
        for t in range(model.order, num_samples):
            for p in range(1, model.order):
                Y[:, t, ep] -= -model.parameters[:, :, p].dot(Y[:, t-p, ep])

        if noise is not None:
            scale = Y.std()
            Y += np.random.randn(*Y.shape) * (scale * noise)

    return Y


def simulate_raw_from_template(sim_samples, bad_segments=None, bad_channels=None, flat_channels=None, noise=None):
    """Simulate raw MEG data from a 306-channel MEGIN template.
    
    Parameters
    ----------
    sim_samples : int
        The number of samples to simulate.
    bad_segments : list of tuples
        The bad segments to simulate.
    bad_channels : list of ints
        The bad channels to simulate.
    flat_channels : list of ints
        The flat channels to simulate.
        
    Returns
    -------
    sim : :py:class:`mne.io.Raw <mne.io.Raw>`
        The simulated data. 
    
    """
    basedir = os.path.dirname(os.path.realpath(__file__))
    basedir = os.path.join(basedir, 'simulation_config')
    info = mne.io.read_info(os.path.join(basedir, 'megin_template_info.fif'))
    with info._unlock():
        info['sfreq'] = 150

    Y = np.zeros((306, sim_samples))
    for mod in ['mag', 'grad']:
        red_model = sails.AbstractLinearModel()
        fname = 'reduced_mvar_params_{0}.npy'.format(mod)
        red_model.parameters = np.load(os.path.join(basedir, fname))
        fname = 'reduced_mvar_residcov_{0}.npy'.format(mod)
        red_model.resid_cov = np.load(os.path.join(basedir, fname))
        red_model.delay_vect = np.arange(20)
        fname = 'reduced_mvar_pcacomp_{0}.npy'.format(mod)
        pcacomp = np.load(os.path.join(basedir, fname))

        Xsim = simulate_data(red_model, num_samples=sim_samples, noise=noise) * 2e-12
        Xsim = pcacomp.T.dot(Xsim[:,:,0])[:,:,None]  # back to full space

        Y[mne.pick_types(info, meg=mod), :] = Xsim[:, :, 0]

    if flat_channels is not None:
        Y[flat_channels, :] = 0

    if bad_channels is not None:
        std = Y[bad_channels, :].std(axis=1)[:, None]
        Y[bad_channels, :] += np.random.randn(len(bad_channels), Y.shape[1]) * std*2

    if bad_segments is not None:
        for seg in bad_segments:
            std = Y[:, seg[0]:seg[1]].std(axis=1)[:, None]
            Y[:, seg[0]:seg[1]] += np.random.randn(Y.shape[0], seg[1]-seg[0]) * std*5

    sim = mne.io.RawArray(Y, info)

    return sim


def simulate_rest_mvar(raw, sim_samples,
                       mvar_pca=32, mvar_order=12,
                       picks=None, modalities=None, drop_dig=False):
    """Simulate resting state data from a raw object using a reduced MVAR model.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        The raw object to simulate from.
    sim_samples : int
        The number of samples to simulate.
    mvar_pca : int
        The number of PCA components to use.
    mvar_order : int
        The MVAR model order.
    picks : dict
        The picks to use. See :py:func:`mne.pick_types <mne.pick_types>`.
    modalities : list of str
        The modalities to use. See :py:func:`mne.pick_types <mne.pick_types>`.
    drop_dig : bool 
        Whether to drop the digitisation points.
        
    Returns
    -------
    sim : :py:class:`mne.io.Raw <mne.io.Raw>`
        The simulated data.
    
    Notes
    -----
    Best used on low sample rate data <200Hz. fiff only for now."""

    if modalities is None:
        modalities = ['mag', 'grad']

    # Fit model and simulate data
    Y = np.zeros((raw.info['nchan'], sim_samples))
    for mod in modalities:
        X = raw.get_data(picks=mod)
        X = X[:, 5000:45000] * 1e12

        red_model, full_model, pca = sails.modelfit.pca_reduced_fit(X, np.arange(mvar_order), mvar_pca)

        scale = X.std() / 1e12
        Xsim = simulate_data(red_model, num_samples=sim_samples) * scale
        Xsim = pca.components.T.dot(Xsim[:,:,0])[:,:,None]  # back to full space

        Y[mne.pick_types(raw.info, meg=mod), :] = Xsim[:, :, 0]

    # Create data info for simulated object
    info = mne.io.anonymize_info(raw.info.copy())
    info['description'] = 'OSL Simulated Dataset'
    info['experimentor'] = 'osl'
    info['proj_name'] = 'osl_simulate'
    info['subject_info'] = {'id': 0, 'first_name': 'OSL', 'last_name': 'Simulated Data'}
    if drop_dig:
        info.pop('dig')

    if picks is None:
        picks = {'meg': True, 'eeg': False,
               'eog': False, 'ecg': False,
               'stim': False, 'misc': False}

    info = mne.pick_info(info, mne.pick_types(info, **pks))

    sim = mne.io.RawArray(Y, info)

    return sim


if __name__ ==  '__main__':

    fname = '/Users/andrew/Projects/ntad/raw_data/meeg_pilots/NTAD_Neo_Pilot2_RSO.fif'
    info = mne.io.read_info(fname)

    info = mne.io.anonymize_info(info)
    info['description'] = 'OSL Simulated Dataset'
    # info['experimentor'] = 'osl'
    # info['proj_name'] = 'osl_simulate'
    info['subject_info'] = {'id': 0, 'first_name': 'OSL', 'last_name': 'Simulated Data'}
    info.pop('dig')

    pks = {'meg': True, 'eeg': False,
           'eog': False, 'ecg': False,
           'stim': False, 'misc': False}

    info = mne.pick_info(info, mne.pick_types(info, **pks))

    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.pick_types(meg=True)
    raw.filter(1,None)  # Remove drifts
    raw.notch_filter(50)  # Remove electrical noise
    raw.resample(150)
    sample_rate = 150

    sim_samples = 10000

    Y = np.zeros((info['nchan'], sim_samples))
    for mod in ['mag', 'grad']:
        X = raw.get_data(picks=mod)
        X = X[:, 5000:45000] * 1e12

        red_model, full_model, pca = sails.modelfit.pca_reduced_fit(X, np.arange(20), 50)

        np.save('reduced_mvar_params_{0}.npy'.format(mod), red_model.parameters)
        np.save('reduced_mvar_residcov_{0}.npy'.format(mod), red_model.resid_cov)
        np.save('reduced_mvar_pcacomp_{0}.npy'.format(mod), pca.components)

        scale = X.std() / 1e12
        Xsim = simulate_data(red_model, num_samples=sim_samples) * scale
        Xsim = pca.components.T.dot(Xsim[:,:,0])[:,:,None]  # back to full space

        Y[mne.pick_types(info, meg=mod), :] = Xsim[:, :, 0]

    sim = mne.io.RawArray(Y, info)
    sim.info['sfreq'] = sample_rate
