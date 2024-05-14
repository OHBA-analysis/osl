"""Save source reconstructed data as numpy files and prepare subset of the dataset.

"""

import os
import mne
import numpy as np
from glob import glob

save_src = False
save_prepared_all = False
save_prepared_rest_all = False
save_prepared_rest_eo = False
save_prepared_rest_ec = False
save_prepared_task_all = False
save_prepared_task_auditory = False
save_prepared_task_nback = True
save_prepared_task_scenes = True
save_prepared_task_sternberg = True
save_prepared_task_verbgeneration = True
save_prepared_task_visuomotor = True

npy_dir = "/well/woolrich/projects/mrc_meguk/all_sites/npy"
fid_dir = f"{npy_dir}/file_ids"
os.makedirs(fid_dir, exist_ok=True)

if save_src:
    os.makedirs(f"{npy_dir}/src", exist_ok=True)

    files = sorted(
        glob("/well/woolrich/projects/mrc_meguk/all_sites/sflip/*/sflip_parc-raw.fif")
    )
    for file in files:
        id = file.split("/")[-2]

        if "noiseAnonymisedTemp" in id or "eyecalibration" in id:
            continue

        raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
        data = raw.pick("misc").get_data(reject_by_annotation="omit", verbose=False).T
        data = data.astype(np.float32)

        filename = f"{npy_dir}/src/{id}.npy"
        print(filename)
        np.save(filename, data)

if save_prepared_all:
    from osl_dynamics.data import Data

    files = sorted(glob(f"{npy_dir}/src/*.npy"))

    with open(f"{fid_dir}/prepared_all.txt", "w") as file:
        for filename in files:
            file.write(f"{filename.split('/')[-1]}\n")

    data = Data(files, load_memmaps=False, n_jobs=16)
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
            "standardize": {},
        }
    )
    data.save(f"{npy_dir}/prepared_all")
    data.delete_dir()

if save_prepared_rest_all:
    from osl_dynamics.data import Data

    files = []
    for task in ["resteyesopen", "resteyesclosed"]:
        files += sorted(glob(f"{npy_dir}/src/*_task-{task}.npy"))

    with open(f"{fid_dir}/prepared_rest_all.txt", "w") as file:
        for filename in files:
            file.write(f"{filename.split('/')[-1]}\n")

    data = Data(files, load_memmaps=False, n_jobs=16)
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
            "standardize": {},
        }
    )
    data.save(f"{npy_dir}/prepared_rest_all")
    data.delete_dir()

if save_prepared_rest_eo:
    from osl_dynamics.data import Data

    files = sorted(glob(f"{npy_dir}/src/*_task-resteyesopen.npy"))

    with open(f"{fid_dir}/prepared_rest_eo.txt", "w") as file:
        for filename in files:
            file.write(f"{filename.split('/')[-1]}\n")

    data = Data(files, load_memmaps=False, n_jobs=16)
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
            "standardize": {},
        }
    )
    data.save(f"{npy_dir}/prepared_rest_eo")
    data.delete_dir()

if save_prepared_rest_ec:
    from osl_dynamics.data import Data

    files = sorted(glob(f"{npy_dir}/src/*_task-resteyesclosed.npy"))

    with open(f"{fid_dir}/prepared_rest_ec.txt", "w") as file:
        for filename in files:
            file.write(f"{filename.split('/')[-1]}\n")

    data = Data(files, load_memmaps=False, n_jobs=16)
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
            "standardize": {},
        }
    )
    data.save(f"{npy_dir}/prepared_rest_ec")
    data.delete_dir()

if save_prepared_task_all:
    from osl_dynamics.data import Data

    files = []
    for task in [
        "auditory", "nback", "scenes", "sternberg", "verbgeneration", "visuomotor"
    ]:
        files += sorted(glob(f"{npy_dir}/src/*_task-{task}*.npy"))

    with open(f"{fid_dir}/prepared_task_all.txt", "w") as file:
        for filename in files:
            file.write(f"{filename.split('/')[-1]}\n")

    data = Data(files, load_memmaps=False, n_jobs=16)
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
            "standardize": {},
        }
    )
    data.save(f"{npy_dir}/prepared_task_all")
    data.delete_dir()

if save_prepared_task_auditory:
    from osl_dynamics.data import Data

    files = sorted(glob(f"{npy_dir}/src/*_task-auditory*.npy"))

    with open(f"{fid_dir}/prepared_task_auditory.txt", "w") as file:
        for filename in files:
            file.write(f"{filename.split('/')[-1]}\n")

    data = Data(files, load_memmaps=False, n_jobs=16)
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
            "standardize": {},
        }
    )
    data.save(f"{npy_dir}/prepared_task_auditory")
    data.delete_dir()

if save_prepared_task_nback:
    from osl_dynamics.data import Data

    files = sorted(glob(f"{npy_dir}/src/*_task-nback.npy"))

    with open(f"{fid_dir}/prepared_task_nback.txt", "w") as file:
        for filename in files:
            file.write(f"{filename.split('/')[-1]}\n")

    data = Data(files, load_memmaps=False, n_jobs=16)
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
            "standardize": {},
        }
    )
    data.save(f"{npy_dir}/prepared_task_nback")
    data.delete_dir()

if save_prepared_task_scenes:
    from osl_dynamics.data import Data

    files = sorted(glob(f"{npy_dir}/src/*_task-scenes.npy"))

    with open(f"{fid_dir}/prepared_task_scenes.txt", "w") as file:
        for filename in files:
            file.write(f"{filename.split('/')[-1]}\n")

    data = Data(files, load_memmaps=False, n_jobs=16)
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
            "standardize": {},
        }
    )
    data.save(f"{npy_dir}/prepared_task_scenes")
    data.delete_dir()

if save_prepared_task_sternberg:
    from osl_dynamics.data import Data

    files = sorted(glob(f"{npy_dir}/src/*_task-sternberg.npy"))

    with open(f"{fid_dir}/prepared_task_sternberg.txt", "w") as file:
        for filename in files:
            file.write(f"{filename.split('/')[-1]}\n")

    data = Data(files, load_memmaps=False, n_jobs=16)
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
            "standardize": {},
        }
    )
    data.save(f"{npy_dir}/prepared_task_sternberg")
    data.delete_dir()

if save_prepared_task_verbgeneration:
    from osl_dynamics.data import Data

    files = sorted(glob(f"{npy_dir}/src/*_task-verbgeneration.npy"))

    with open(f"{fid_dir}/prepared_task_verbgeneration.txt", "w") as file:
        for filename in files:
            file.write(f"{filename.split('/')[-1]}\n")

    data = Data(files, load_memmaps=False, n_jobs=16)
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
            "standardize": {},
        }
    )
    data.save(f"{npy_dir}/prepared_task_verbgeneration")
    data.delete_dir()

if save_prepared_task_visuomotor:
    from osl_dynamics.data import Data

    files = sorted(glob(f"{npy_dir}/src/*_task-visuomotor.npy"))

    with open(f"{fid_dir}/prepared_task_visuomotor.txt", "w") as file:
        for filename in files:
            file.write(f"{filename.split('/')[-1]}\n")

    data = Data(files, load_memmaps=False, n_jobs=16)
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
            "standardize": {},
        }
    )
    data.save(f"{npy_dir}/prepared_task_visuomotor")
    data.delete_dir()
