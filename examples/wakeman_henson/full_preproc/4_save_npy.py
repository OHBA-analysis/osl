"""Save training data as npy files.

"""

from glob import glob
from osl_dynamics.data import Data

files = sorted(glob("/well/woolrich/projects/wakeman_henson/spring23/src/*/sflip_parc-raw.fif"))
data = Data(files, picks="misc", reject_by_annotation="omit")
data.save("/well/woolrich/projects/wakeman_henson/spring23/training")
data.delete_dir()
