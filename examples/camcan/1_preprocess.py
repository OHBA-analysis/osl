"""The script used to preproces the CamCAN data in BMRC.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import pathlib
from glob import glob
from dask.distributed import Client

from osl import preprocessing, utils

BASE_DIR = "/well/woolrich/projects/camcan"
RAW_DIR = BASE_DIR + "/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002"
PREPROC_DIR = BASE_DIR + "/autumn22/preproc"

config = """
    preproc:
    - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}
    - bad_segments: {segment_len: 500, picks: mag}
    - bad_segments: {segment_len: 500, picks: mag, mode: diff}
    - bad_segments: {segment_len: 1000, picks: mag}
    - bad_segments: {segment_len: 1000, picks: mag, mode: diff}
    - bad_segments: {segment_len: 500, picks: grad}
    - bad_segments: {segment_len: 500, picks: grad, mode: diff}
    - bad_segments: {segment_len: 1000, picks: grad}
    - bad_segments: {segment_len: 1000, picks: grad, mode: diff}
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Get input files
    inputs = []
    for subject in glob(RAW_DIR + "/sub-*"):
        subject = pathlib.Path(subject).stem
        inputs.append(RAW_DIR + f"/{subject}/mf2pt2_{subject}_ses-rest_task-rest_meg.fif")

    # Setup parallel processing
    client = Client(n_workers=16, threads_per_worker=1)

    # Main preprocessing
    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=PREPROC_DIR,
        overwrite=True,
        dask_client=True,
    )
