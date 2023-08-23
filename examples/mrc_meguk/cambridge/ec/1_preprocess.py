"""Preprocessing.

"""
from glob import glob
from pathlib import Path
from dask.distributed import Client

from osl import preprocessing, utils

RAW_DIR = "/well/woolrich/projects/mrc_meguk/raw/Cambridge/derivatives"
RAW_FILE = RAW_DIR + "/{0}/meg/{0}_task-resteyesclosed_proc-sss_meg.fif"
PREPROC_DIR = "/well/woolrich/projects/mrc_meguk/cambridge/ec/preproc"

config = """
    preproc:
    - pick: {picks: [meg, eog, ecg]}
    - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}
    - bad_segments: {segment_len: 500, picks: meg, significance_level: 0.1}
    - bad_segments: {segment_len: 500, picks: meg, mode: diff, significance_level: 0.1}
    - bad_channels: {picks: meg, significance_level: 0.1}
    - ica_raw: {picks: meg, n_components: 64}
    - ica_autoreject: {picks: meg, ecgmethod: correlation, eogthreshold: auto}
    - interpolate_bads: {}
"""
if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    inputs = []
    for directory in sorted(glob(RAW_DIR + "/sub-*")):
        subject = Path(directory).name
        raw_file = RAW_FILE.format(subject)
        if Path(raw_file).exists():
            inputs.append(raw_file)

    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=PREPROC_DIR,
        overwrite=True,
        dask_client=True,
    )
