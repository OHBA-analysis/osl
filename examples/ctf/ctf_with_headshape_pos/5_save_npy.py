"""Save sign flipped parcel data as numpy files.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from glob import glob

from osl_dynamics.data import Data

files = sorted(glob("data/src/*/sflip_parc-raw.fif"))
data = Data(
    files,
    picks="misc",
    reject_by_annotation="omit",
    n_jobs=4,
)
data.save("data/npy")
data.delete_dir()
