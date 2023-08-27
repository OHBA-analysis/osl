"""Data preparation: time-delay embedding and principal component analysis.

"""

from glob import glob

from osl_dynamics.data import Data

files = sorted(glob("/well/woolrich/projects/camcan/summer23/src/*/sflip_parc-raw.fif"))
data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=16)
methods = {
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 100},
    "standardize": {},
}
data.prepare(methods)
data.save("/well/woolrich/projects/camcan/summer23/prepared")
data.delete_dir()
