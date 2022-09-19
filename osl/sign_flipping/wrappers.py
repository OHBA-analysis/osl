"""Wrapper functions for sign flipping.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from pathlib import Path

from .sign_flipping import (
    load_covariances,
    find_template_subject,
    find_flips,
    plot_sign_flipping,
    apply_flips,
)
from ..utils import validate_outdir


def fix_sign_ambiguity(
    subject_files,
    n_embeddings=1,
    standardize=True,
    n_init=1,
    n_iter=100,
    max_flips=20,
    outdir=None,
):
    """Wrapper function for fixing the dipole sign ambiguity.

    Parameters
    ----------
    subject_files : list of str
        List of paths to parcellated data files.
    n_embeddings : int
        Number of time-delay embeddings that we will use (if we are doing any).
    standardize : bool
        Should we standardize (z-transform) the data before sign flipping?
    n_init : int
        Number of initializations.
    n_iter : int
        Number of sign flipping iterations per subject to perform.
    max_flips : int
        Maximum number of channels to flip in an iteration.
    outdir : str
        Directory to write sign flipped data to.
    """
    print("Performing sign flipping:")

    # Validation
    n_subjects = len(subject_files)
    if n_subjects < 2:
        raise ValueError(f"two or more subject files must be passed, got {n_subjects}")

    if outdir is None:
        # Use the directory the parcellated data files are kept in
        outdir = Path(subject_files[0]).parent + "/sflip"
    outdir = validate_outdir(outdir)
    print("Using output directory:", outdir)

    # Calculate the covariance matrix of each subject
    covs = load_covariances(subject_files, n_embeddings, standardize)

    # Find a subject to use as a template
    template_index = find_template_subject(covs, n_embeddings)
    print("Using template:", subject_files[template_index])

    # Find the channels to flip
    flips, metrics = find_flips(
        covs, template_index, n_embeddings, n_init, n_iter, max_flips
    )

    # Plot a summary figure describing the sign flipping solution
    plot_sign_flipping(covs, n_embeddings, template_index, metrics, outdir)

    # Apply flips to the parcellated data
    apply_flips(subject_files, flips, outdir)
