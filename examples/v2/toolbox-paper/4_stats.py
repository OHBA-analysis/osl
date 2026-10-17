import os
import numpy as np
import glmtools
import matplotlib.pyplot as plt
from dask.distributed import Client
from osl_ephys import preprocessing, glm

    
if __name__ == "__main__":
    client = Client(n_workers=16, threads_per_worker=1)
  
    config = """
      preproc:
        - read_dataset: {ftype: sflip_parc-raw}
        - epochs: {picks: misc, tmin: -0.2, tmax: 0.3}
        - glm_add_regressor: {name: famous, rtype: Categorical, codes: [5 6 7]}
        - glm_add_regressor: {name: unfamiliar, rtype: Categorical, codes: [13 14 15]}
        - glm_add_regressor: {name: scrambled, rtype: Categorical, codes: [17 18 19]}
        - glm_add_contrast: {name: Mean, values: {famous: 1/3, unfamiliar: 1/3, scrambled: 1/3}}
        - glm_add_contrast: {name: Faces-Scrambled, values: {famous: 1, unfamiliar: 1, scrambled: -2}}
        - glm_fit: {target: epochs, method: glm_epochs}
      group:
        - glm_add_regressor: {name: Subject, rtype: Categorical, key: Subject, codes: unique}
        - glm_add_contrast: {name: Mean, values: unique, key: Subject}
        - glm_fit: {method: epochs, tmin: 0.05, tmax: 0.3}
        - glm_permutations: {method: epochs, target: group_glm, contrast: Mean, type: max, nperms: 1000, threshold: 0.99}
    """    
    proc_dir = "ds117/processed"
    src_files = sorted(utils.Study(os.path.join(proc_dir, 
"sub{sub_id}-run{run_id}", "sub{sub_id}-run{run_id}_sflip_parc-raw.fif")).get())    
    subjects = [f"sub{i+1:03d}-run{j+1:02d}" for i in range(19) for j in range(6)]
    covs = [f"Subject": [sub.split("-")[0]for sub in subjects]

    preprocessing.run_proc_batch(
        config,
        src_files,
        subjects,
        outdir=proc_dir,
        ftype='raw',
        covs=covs,
        dask_client=True,
        overwrite=True,
        gen_report=False,
        skip_save=['events', 'raw', 'ica', 'event_id', 'sflip_parc-raw'],
    )

