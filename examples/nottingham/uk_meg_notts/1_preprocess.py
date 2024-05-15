"""Preprocess uk_meg_notts data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from glob import glob
from pathlib import Path
from dask.distributed import Client

from osl import preprocessing, utils

BASE_DIR = "/well/woolrich/projects/uk_meg_notts/eo/oslpy22"
RAW_DIR = BASE_DIR + "/raw"
RAW_FILE = RAW_DIR + "/{0}/{0}_raw.fif"
PREPROC_DIR = BASE_DIR + "/preproc"

config = """
    preproc:
    - filter: {l_freq: 0.5, h_freq: 125, method: 'iir', iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}
    - bad_segments: {segment_len: 500, picks: 'meg'}
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    inputs = []
    for directory in sorted(glob(RAW_DIR + "/30*")):
        subject = Path(directory).name
        inputs.append(RAW_FILE.format(subject))

    client = Client(n_workers=16, threads_per_worker=1)

    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=PREPROC_DIR,
        overwrite=True,
        gen_report=False,
        dask_client=True,
    )
