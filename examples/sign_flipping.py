"""Standalone script for peforming sign flipping with MATLAB data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
from scipy import io

from osl.source_recon.sign_flipping import (
    load_covariances,
    find_template_subject,
    find_flips,
    apply_flips,
)

SRC_DIR = "/ohba/pi/mwoolrich/cgohil/uk_meg_notts/bmrc_data"

N_EMBEDDINGS = 15
STANDARDIZE = True
N_INIT = 2
N_ITER = 2500
MAX_FLIPS = 20

# Input data
subject_files = []
for i in range(1, 11):
    subject_files.append(SRC_DIR + f"/subject{i}.mat")


def load_matlab(filename):
    """Function to load data files."""
    data = io.loadmat(filename)
    return data["X"]


def save_matlab(filename, X):
    """Function to save data files."""
    T = X.shape[0]
    io.savemat(filename, {"X": X, "T": T})


# Get covariance matrices
covs = load_covariances(
    subject_files,
    N_EMBEDDINGS,
    STANDARDIZE,
    loader=load_matlab,
)

# Find a subject to use as a template
template = find_template_subject(covs, N_EMBEDDINGS)
print("Using template:", subject_files[template])

# Loop through each subject
for i in range(len(subject_files)):
    print("Subject", i + 1)

    if i == template:
        # Don't need to do sign flipping on the template subject
        parc_data = load_matlab(subject_files[i])
        save_matlab(SRC_DIR + f"/sflip{i + 1}.mat", parc_data)
        continue

    # Find the channels to flip
    flips, metrics = find_flips(
        covs[i], covs[template], N_EMBEDDINGS, N_INIT, N_ITER, MAX_FLIPS
    )

    # Apply flips to the parcellated data and save
    parc_data = load_matlab(subject_files[i])
    parc_data *= flips[np.newaxis, ...]
    save_matlab(SRC_DIR + f"/sflip{i + 1}.mat", parc_data)
