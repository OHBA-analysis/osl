"""Performs sign flipping.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from glob import glob

from osl.source_recon import find_template_subject, run_src_batch


# Source reconstruction directory
src_dir = "/ohba/pi/knobre/cgohil/covid/src"

# Subjects to sign flip
# We create a list by looking for subjects that have a rhino/parc.npy file
subjects = []
for path in sorted(glob(src_dir + "/*/rhino/parc.npy")):
    subject = path.split("/")[-3]
    subjects.append(subject)

# Find a good template subject to align other subjects to
template = find_template_subject(
    src_dir, subjects, n_embeddings=15, standardize=True
)

# Settings for batch processing
config = f"""
    source_recon:
    - fix_sign_ambiguity:
        template: {template}
        n_embeddings: 15
        standardize: True
        n_init: 3
        n_iter: 500
        max_flips: 20
"""

# Do the sign flipping
run_src_batch(config, src_dir, subjects)
