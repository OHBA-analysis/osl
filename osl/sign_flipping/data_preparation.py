"""Functions used to prepare data for training a model.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np


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


def standardize(x):
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
