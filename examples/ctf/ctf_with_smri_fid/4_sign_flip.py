"""Align the sign of each parcel time course across subjects
and save the data as a vanilla numpy file.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import mne
import numpy as np
from glob import glob

from osl.source_recon.sign_flipping import (
    load_covariances,
    find_template_subject,
    find_flips,
    apply_flips,
)

def load(filename):
    """Load data without bad segments."""
    raw = mne.io.read_raw_fif(filename, verbose=False)
    raw = raw.pick("misc")
    data = raw.get_data(reject_by_annotation="omit", verbose=False)
    return data.T


# Files to sign flip
files = sorted(glob("data/src/*/parc/parc-raw.fif"))

# Get covariance matrices
covs = load_covariances(
    files,
    n_embeddings=15,
    standardize=True,
    loader=load,
)

# Load template covariance
template_cov = np.load("../camcan_norm_model/template_cov.npy")

# Output directory
os.makedirs("data/npy", exist_ok=True)

# Do sign flipping
for i in range(len(files)):
    print("Sign flipping", files[i])

    # Find the channels to flip
    flips, metrics = find_flips(
        covs[i],
        template_cov,
        n_embeddings=15,
        n_init=3,
        n_iter=2500,
        max_flips=20,
    )

    # Apply flips to the parcellated data and save
    parc_data = load(files[i])
    parc_data *= flips[np.newaxis, ...]
    subject = files[i].split("/")[-3]
    np.save(f"data/npy/{subject}.npy", parc_data)
