"""Functions for fixing the dipole sign ambiguity of beamformed data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

from . import data_preparation


def find_flips(covs, template_index, n_embeddings, n_init=1, n_iter=100, max_flips=20):
    """Find channels to flip.

    We search for the channels to flip by randomly flipping them and saving the
    flips that maximise the correlation of the covariance matrices between subjects.

    Parameters
    ----------
    covs : numpy.ndarray
        Covariance matrices of time-delay embedded and standardized data.
        Shape must be
        (n_subjects, n_channels*n_embeddings, n_channels*n_embeddings).
    template_index : int
        Index for the template subject.
    n_embeddings : int
        Number of time delay embeddings used.
    n_init : int
        Number of initializations.
    n_iter : int
        Number of sign flipping iterations per subject to perform.
    max_flips : int
        Maximum number of channels to flip in an iteration.

    Returns
    -------
    best_flips : numpy.ndarray
        A (n_subjects, n_channels) array of 1s and -1s indicating whether or not
        to flip a channels.
    metric : numpy.ndarray
        Evaluation metric (correlation between covariance matrices) as a function
        of iterations. Shape is (n_subject, n_iter).
    """
    # Get the number of subjects and channels
    n_subjects = covs.shape[0]
    n_channels = covs.shape[-1] // n_embeddings

    # Validation
    if max_flips > n_channels:
        raise ValueError(
            f"max_flips must be less than the number of channels ({n_channels})"
        )

    # Covariance matrix we're trying to match the other subjects to
    template_cov = covs[template_index]

    # Find the best channels to flip
    best_flips = np.ones([n_subjects, n_channels])
    best_metrics = np.zeros(n_subjects)
    metrics = []
    for n in trange(n_init, desc="Sign flipping", ncols=98):

        # Loop through subjects to align to the template
        for i in trange(n_subjects, desc="Aligning dipoles", ncols=98):
            if i == template_index:
                # Skip the template subject
                best_metrics[i] = 1
                continue

            # Reset the flips and calculate the evaluation metric before sign flipping
            flips = np.ones(n_channels)
            metric = covariance_matrix_correlation(covs[i], template_cov, n_embeddings)

            # Randomly permute the sign of different channels and calculate the metric
            for j in range(n_iter):
                new_flips = randomly_flip(flips, max_flips)
                cov = apply_flips_to_covariance(covs[i], new_flips, n_embeddings)
                new_metric = covariance_matrix_correlation(
                    cov, template_cov, n_embeddings
                )
                if new_metric > metric:
                    # We've found an improved solution, let's save it
                    flips = new_flips
                    metric = new_metric

            # Update best_flips if this was the best init
            if metric > best_metrics[i]:
                best_flips[i] = flips
                best_metrics[i] = metric

        # Save metrics as a function of init
        metrics.append(np.copy(best_metrics))

    return best_flips, metrics


def load_covariances(subject_files, n_embeddings=1, standardize=True):
    """Loads data and returns its covariance matrix.

    Parameters
    ----------
    subject_files : list of str
        List of paths to parcellated data files to load.
    n_embeddings : int
        Number of time-delay embeddings to perform.
    standardize : bool
        Should we standardize the data?

    Returns
    -------
    covs : numpy.ndarray
        Covariance matrices.
    """
    covs = []
    for i in trange(len(subject_files), desc="Calculating covariances", ncols=98):
        # Load data
        x = np.load(subject_files[i])

        # Prepare
        x = data_preparation.time_embed(x, n_embeddings)
        if standardize:
            x = data_preparation.standardize(x)

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
    median = np.median(metric_sum)

    return np.argwhere(metric_sum == median)[0][0]


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
    n_channels = flips.shape[0]
    new_flips = np.copy(flips)

    # Select the number of channels to flip
    n_channels_to_flip = np.random.choice(max_flips, size=1)

    # Select the channels to flip
    random_channels_to_flip = np.random.choice(
        n_channels, size=n_channels_to_flip, replace=False
    )
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


def apply_flips(subject_files, flips, outdir):
    """Saves the sign flipped data.

    Parameters
    ----------
    subject_files : list of str
        List of paths to parcellated data files.
    flips : numpy.ndarray
        Flips to apply.
    outdir : str
        Path to output directory. We will save npy files containing the sign
        flipped data to this directory.
    """
    # Validation
    n_subjects = len(subject_files)
    n_flips = len(flips)
    if n_subjects != n_flips:
        raise ValueError(
            f"different number of subject files ({n_subjects}) and flips ({n_flips}) passed"
        )

    for i in trange(n_subjects, desc="Saving data", ncols=98):
        # Load parcellated data
        data = np.load(subject_files[i])

        # Flip the sign of the channels
        flipped_data = data * flips[i][np.newaxis, ...]

        # Save
        outfile = Path(outdir) / Path(subject_files[i]).name
        np.save(outfile, flipped_data)


def plot_sign_flipping(covs, n_embeddings, template_index, metrics, outdir):
    """Plots the results of the sign flipping.

    Parameters
    ----------
    covs : numpy.ndarray
        Covariance matrices of time-delay embedded and standardized data.
        Shape must be
        (n_subjects, n_channels*n_embeddings, n_channels*n_embeddings).
    n_embeddings : int
        Number of time delay embeddings used.
    template_index : int
        Index for the template subject.
    metrics : numpy.ndarray
        Metrics calculated during sign flipping.
    outdir : str
        Output directory to save plot to.
    """
    plot_filename = Path(outdir) / "sign_flipping_results.png"
    print("Saving:", plot_filename)

    # Create a figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    # Plot the covariance of the template subject
    i, j = np.triu_indices(covs.shape[-1], k=n_embeddings)
    template_cov = np.zeros([covs.shape[-1], covs.shape[-1]])
    template_cov[i, j] = covs[template_index, i, j]
    template_cov[j, i] = covs[template_index, j, i]

    im = ax[0].imshow(template_cov)
    ax[0].set_title("Template subject covariance")

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Plot the final correlation with respect to the template subject
    old_metric = []
    for cov in covs:
        old_metric.append(
            covariance_matrix_correlation(template_cov, cov, n_embeddings)
        )
    ax[1].plot(old_metric, label="Before")
    for i, metric in enumerate(metrics):
        ax[1].plot(metric, label=f"After (init {i + 1})")
    ax[1].legend()
    ax[1].set_xlabel("Subject")
    ax[1].set_ylabel("Correlation")

    # Save plot
    plt.tight_layout()
    fig.savefig(plot_filename)
    fig.clf()
