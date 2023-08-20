"""Performs sign flipping on epoched parcellated data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from glob import glob

from osl.source_recon import find_template_subject, run_src_batch

# Setup paths to epoched data files
event_type = "internal_disp"
src_dir = f"/ohba/pi/knobre/cgohil/int_ext/src/{event_type}"

# Subjects to sign flip
# We create a list by looking for subjects that have a parc/parc-epo.fif file
subjects = []
for path in sorted(glob(src_dir + "/*/parc/parc-epo.fif")):
    subject = path.split("/")[-3]
    subjects.append(subject)

# Find a good template subject to align other subjects to
template = find_template_subject(
    src_dir, subjects, n_embeddings=15, standardize=True, epoched=True
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
        epoched: true
"""

# Do the sign flipping
run_src_batch(config, src_dir, subjects)
