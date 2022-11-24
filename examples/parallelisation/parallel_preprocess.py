"""Example script for preprocessing CamCAN in parallel.

In this script the preprocessing for multiple fif files will be done in parallel.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import pathlib
from glob import glob
from dask.distributed import Client

from osl import preprocessing, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    raw_dir = "/ohba/pi/mwoolrich/datasets/CamCan_2021/cc700/meg/pipeline/release005/BIDSsep/rest"
    preproc_dir = "/ohba/pi/mwoolrich/cgohil/camcan/preproc"

    config = """
        preproc:
        - filter: {l_freq: 0.5, h_freq: 125, method: 'iir', iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100 150 200}
        - resample: {sfreq: 250}
        - bad_channels: {picks: 'mag'}
        - bad_channels: {picks: 'grad'}        
        - bad_segments: {segment_len: 2000, picks: 'mag'}
        - bad_segments: {segment_len: 2000, picks: 'grad'}
    """

    # Get input files
    inputs = []
    for subject in glob(raw_dir + "/sub-*"):
        subject = pathlib.Path(subject).stem
        inputs.append(raw_dir + f"/{subject}/ses-rest/meg/{subject}_ses-rest_task-rest_meg.fif")
    inputs = inputs[:2]

    # Setup a Dask client for parallel processing
    #
    # Generally, we advise leaving threads_per_worker=1
    # and setting n_workers to the number of CPUs you want
    # to use.
    #
    # Note, we recommend you do not set n_workers to be
    # greater than half the total number of CPUs you have.
    # Also, each worker will process a separate fif file
    # so setting n_workers greater than the number of fif
    # files you want to process won't speed up the script.
    client = Client(n_workers=2, threads_per_worker=1)

    # Main preprocessing
    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=preproc_dir,
        overwrite=True,
        dask_client=True,
    )
