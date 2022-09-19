"""Performs sign flipping.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from glob import glob

from osl import sign_flipping


SRC_DIR = "/ohba/pi/mwoolrich/cgohil/ukmp_notts/src"
SF_DIR = "/ohba/pi/mwoolrich/cgohil/ukmp_notts/sflip"

# Get parcellated data files
parc_files = sorted(glob(SRC_DIR + "/sub-*.npy"))

# Fix the dipole sign ambiguity
sign_flipping.fix_sign_ambiguity(
    parc_files, n_embeddings=15, n_iter=500, outdir=SF_DIR
)
