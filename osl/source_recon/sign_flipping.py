"""Functions for fixing the dipole sign ambiguity of beamformed data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import pickle
import os
import os.path as op
from pathlib import Path

import mne.io

from . import parcellation

import numpy as np
from tqdm import trange

from osl.utils.logger import log_or_print


def find_flips(
    cov, template_cov, n_embeddings, n_init, n_iter, max_flips, use_tqdm=True
):
    """Find channels to flip.

    We search for the channels to flip by randomly flipping them and saving the
    flips that maximise the correlation of the covariance matrices between subjects.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
    template_cov : numpy.ndarray
        Template covariance matrix.
    n_embeddings : int
        Number of time-delay embeddings.
    n_init : int
        Number of initializations.
    n_iter : int
        Number of sign flipping iterations per subject to perform.
    max_flips : int
        Maximum number of channels to flip in an iteration.
    use_tqdm : bool
        Should we display a tqdm progress bar?

    Returns
    -------
    best_flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s indicating whether or not
        to flip a channels.
    metrics : numpy.ndarray
        Evaluation metric (correlation between covariance matrices) as a function
        of iterations. Shape is (n_iter + 1,).
    """
    log_or_print("find_flips")

    # Get the number of channels
    n_channels = cov.shape[-1] // n_embeddings

    # Validation
    if max_flips > n_channels:
        raise ValueError(
            f"max_flips ({max_flips}) must be less than the number of channels ({n_channels})"
        )

    # Find the best channels to flip
    best_flips = np.ones(n_channels)
    best_metric = 0
    metrics = []
    for n in range(n_init):

        # Reset the flips and calculate the evaluation metric before sign flipping
        flips = np.ones(n_channels)
        metric = covariance_matrix_correlation(cov, template_cov, n_embeddings)
        if n == 0:
            metrics.append(metric)
            log_or_print(f"init {n}, unflipped metric: {metric}")

        # Randomly permute the sign of different channels and calculate the metric
        if use_tqdm:
            iterator = trange(n_iter, desc="sign flipping", ncols=98)
        else:
            iterator = range(n_iter)
        for j in iterator:
            new_flips = randomly_flip(flips, max_flips)
            new_cov = apply_flips_to_covariance(cov, new_flips, n_embeddings)
            new_metric = covariance_matrix_correlation(
                new_cov, template_cov, n_embeddings
            )
            if new_metric > metric:
                # We've found an improved solution, let's save it
                flips = new_flips
                metric = new_metric

        # Update best_flips if this was the best init
        if metric > best_metric:
            best_flips = flips
            best_metric = metric

        # Save metric as a function of init
        metrics.append(best_metric)
        log_or_print(f"init {n}, current best metric: {best_metric}")

    return best_flips, metrics


def load_covariances(
    parc_files, n_embeddings=1, standardize=True, loader=np.load, use_tqdm=True
):
    """Loads data and returns its covariance matrix.

    Parameters
    ----------
    parc_files : list of str
        List of paths to parcellated data files to load.
    n_embeddings : int
        Number of time-delay embeddings to perform.
    standardize : bool
        Should we standardize the data?
    loader : function
        Custom function to load parcellated data files.
    use_tqdm : bool
        Should we display a tqdm progress bar?

    Returns
    -------
    covs : numpy.ndarray
        Covariance matrices.
    """
    covs = []
    if use_tqdm:
        iterator = trange(len(parc_files), desc="Calculating covariances", ncols=98)
    else:
        iterator = range(len(parc_files))
    for i in iterator:
        # Load data
        x = loader(parc_files[i])

        # Prepare
        x = time_embed(x, n_embeddings)
        if standardize:
            x = std_data(x)

        # Calculate the covariance
        covs.append(np.cov(x, rowvar=False))

    return np.array(covs)


def find_template_subject(covs, diag_offset=0):
    """Find a good template subject to use to align dipoles.

    We select the median subject after calculating the similarity between
    the covariances of each subject.

    Parameters
    ----------
    covs : numpy.ndarray
        Covariance of each subject.
        Shape much be (n_subjects, n_channels, n_channels).
    diag_offset : int
        Offset to apply when getting the upper triangle of the covariance matrix
        before calculating the correlation between covariances.

    Returns
    -------
    index : int
        Index for the template subject.
    """
    # Calculate the similarity between subjects
    n_subjects = len(covs)
    metric = np.zeros([n_subjects, n_subjects])
    for i in trange(n_subjects, desc="Comparing subjects", ncols=98):
        for j in range(i + 1, n_subjects):
            metric[i, j] = covariance_matrix_correlation(
                covs[i], covs[j], diag_offset, mode="abs"
            )
            metric[j, i] = metric[i, j]

    # Get the median subject
    metric_sum = np.sum(metric, axis=1)
    argmedian = np.argsort(metric_sum)[len(metric_sum) // 2]

    return argmedian


def covariance_matrix_correlation(M1, M2, diag_offset=0, mode=None):
    """Calculates the Pearson correlation between covariance matrices.

    Parameters
    ----------
    M1 : numpy.ndarray
        First covariance matrix.
    M2 : numpy.ndarray
        Second covariance matrix.
    diag_offset : int
        To calculate the distance we take the upper triangle.
        This argument allows us to specify an offet from the diagonal
        so we can choose not to take elements near the diagonal.
    mode : str
        Either 'abs', 'sign' or None.
    """
    if mode == "abs":
        M1 = np.abs(M1)
        M2 = np.abs(M2)
    elif mode == "sign":
        M1 = np.sign(M1)
        M2 = np.sign(M2)

    # Get the upper triangles
    i, j = np.triu_indices(M1.shape[0], k=diag_offset)
    M1 = M1[i, j]
    M2 = M2[i, j]

    # Calculate correlation
    return np.corrcoef([M1, M2])[0, 1]


def randomly_flip(flips, max_flips):
    """Randomly flips some channels.

    Parameters
    ----------
    flips : numpy.ndarray
        Vector of 1s and -1s indicating which channels to flip.
    max_flips : int
        Maximum number of channels to change in this function.

    Returns
    -------
    new_flips : numpy.ndarray
        Vector of 1s and -1s indicating which channels to flip.
    """

    # Select the number of channels to flip
    n_channels_to_flip = np.random.choice(max_flips, size=1)

    # Select the channels to flip
    n_channels = flips.shape[0]
    random_channels_to_flip = np.random.choice(
        n_channels, size=n_channels_to_flip, replace=False
    )
    new_flips = np.copy(flips)
    new_flips[random_channels_to_flip] *= -1

    return new_flips


def apply_flips_to_covariance(cov, flips, n_embeddings=1):
    """Applies flips to a covariance matrix.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix to apply flips to. Shape must be
        (n_channels*n_embeddings, n_channels*n_embeddings).
    flips : numpy.ndarray
        Vector of 1s and -1s indicating whether or not to flip
        a channels. Shape must be (n_channels,).
    n_embeddings : int
        Number of embeddings used when calculating the covariance.

    Returns
    -------
    cov : numpy.ndarray
        Flipped covariance matrix.
    """
    # flips is a (n_channels,) array however the covariance matrix is
    # (n_channels*n_embeddings, n_channels*n_embeddings), we repeat the
    # flips vector to account for the extra channels due to time embedding
    flips = np.repeat(flips, n_embeddings)[np.newaxis, ...]
    flips = flips.T @ flips
    return cov * flips


def apply_flips(src_dir, subject, flips):
    """Saves the sign flipped data.

    Parameters
    ----------
    src_dir : str
        Path to source reconstruction directory.
    subject : str
        Subject name/id.
    flips : numpy.ndarray
        Flips to apply.
    """
    # Load parcellated data
    parc_file = op.join(src_dir, str(subject), "rhino", "parc.npy")
    data = np.load(parc_file)

    # Flip the sign of the channels
    flipped_data = data * flips[np.newaxis, ...]

    # Save
    outfile = op.join(src_dir, str(subject), "sflip_parc.npy")
    log_or_print(f"saving: {outfile}")
    np.save(outfile, flipped_data)

    # Create and save mne raw object for the sign flipped parcellated data
    parc_fif_file = op.join(src_dir, str(subject), "rhino", "parc-raw.fif")
    parc_raw = mne.io.read_raw_fif(parc_fif_file)
    sf_parc_fif_file = op.join(src_dir, str(subject), "sflip_parc-raw.fif")
    sf_parc_raw = parcellation.convert2mne_raw(flipped_data, parc_raw)
    sf_parc_raw.save(sf_parc_fif_file, overwrite=True)

def time_embed(x, n_embeddings):
    """Performs time-delay embedding.

    Parameters
    ----------
    x : numpy.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    n_embeddings : int
        Number of samples in which to shift the data. Must be an odd number.

    Returns
    -------
    sliding_window_view
        Time embedded data. Shape is (n_samples, n_channels * n_embeddings).
    """
    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    te_shape = (
        x.shape[0] - (n_embeddings - 1),
        x.shape[1] * n_embeddings,
    )

    return (
        np.lib.stride_tricks.sliding_window_view(x=x, window_shape=te_shape[0], axis=0)
        .T[..., ::-1]
        .reshape(te_shape)
    )


def std_data(x):
    """Standardize (z-transform) the data.

    Parameters
    ----------
    x : numpy.ndarray
        Data. Shape must be (n_samples, n_channels).

    Returns
    -------
    std_x: numpy.ndarray
        Standardized time series.
    """
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)
