"""Preprocessing.

"""

from dask.distributed import Client

from osl import preprocessing, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=6, threads_per_worker=1)

    config = """
        preproc:
        - set_channel_types: {EEG061: eog, EEG062: eog, EEG063: ecg}
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag}
        - bad_segments: {segment_len: 500, picks: grad}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff}
        - bad_segments: {segment_len: 500, picks: grad, mode: diff}
        - ica_raw: {picks: meg, n_components: 40}
        - ica_autoreject: {picks: meg, ecgmethod: correlation, eogthreshold: auto}
        - interpolate_bads: {}
    """

    raw_dir = "/well/woolrich/projects/wakeman_henson/ds117"
    for sub in range(1, 20):
        inputs = []
        for run in range(1, 7):
            inputs.append(f"{raw_dir}/sub{sub:03d}/MEG/run_{run:02d}_sss.fif")
        preproc_dir = f"/well/woolrich/projects/wakeman_henson/spring23/preproc/sub{sub:02d}"
        preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=preproc_dir,
            overwrite=True,
            dask_client=True,
        )
