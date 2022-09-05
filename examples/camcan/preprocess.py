"""Example script for preprocessing CamCAN.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import pathlib
from glob import glob
from dask.distributed import Client

from osl import preprocessing, utils

RAW_DIR = "/ohba/pi/mwoolrich/datasets/CamCan_2021/cc700/meg/pipeline/release005/BIDSsep/rest"
PREPROC_DIR = "/ohba/pi/mwoolrich/cgohil/camcan/preproc"

config = """
    preproc:
    - filter: {l_freq: 0.5, h_freq: 125, method: 'iir', iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100 150 200}
    - resample: {sfreq: 250}
    - bad_segments: {segment_len: 2000, picks: 'mag'}
    - bad_segments: {segment_len: 2000, picks: 'grad'}
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Get input files
    inputs = []
    for subject in glob(RAW_DIR + "/sub-*"):
        subject = pathlib.Path(subject).stem
        inputs.append(RAW_DIR + f"/{subject}/ses-rest/meg/{subject}_ses-rest_task-rest_meg.fif")
    inputs = inputs[:2]

    # Setup parallel processing
    client = Client(n_workers=4, threads_per_worker=1)

    # Main preprocessing
    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=PREPROC_DIR,
        overwrite=True,
        dask_client=True,
    )
