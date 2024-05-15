"""Preprocessing.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from glob import glob
from dask.distributed import Client

from osl import preprocessing, utils

# Elekta
do_oxford = False
do_cambridge = False

# CTF
do_nottingham = False
do_cardiff = True

# ??
do_glasgow = False

raw_dir = "/well/woolrich/projects/mrc_meguk/raw"
preproc_dir = "/well/woolwich/projects/mrc_meguk/all_sites/preproc"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    if do_oxford:
        config = """
            preproc:
            - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
            - notch_filter: {freqs: 50 100, notch_widths: 2}
            - resample: {sfreq: 250}
            - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
            - bad_segments: {segment_len: 500, picks: grad, significance_level: 0.1}
            - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
            - bad_segments: {segment_len: 500, picks: grad, mode: diff, significance_level: 0.1}
            - bad_channels: {picks: mag, significance_level: 0.1}
            - bad_channels: {picks: grad, significance_level: 0.1}
            - ica_raw: {picks: meg, n_components: 64}
            - ica_autoreject: {picks: mag, ecgmethod: correlation, eogthreshold: auto}
            - interpolate_bads: {}
        """

        inputs = sorted(glob(f"{raw_dir}/Oxford/derivatives/*/meg/*.fif"))
        outdir = f"{preproc_dir}/Oxford"

        preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=outdir,
            overwrite=True,
            dask_client=True,
        )

    if do_cambridge:
        config = """
            preproc:
            - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
            - notch_filter: {freqs: 50 100, notch_widths: 2}
            - resample: {sfreq: 250}
            - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
            - bad_segments: {segment_len: 500, picks: grad, significance_level: 0.1}
            - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
            - bad_segments: {segment_len: 500, picks: grad, mode: diff, significance_level: 0.1}
            - bad_channels: {picks: mag, significance_level: 0.1}
            - bad_channels: {picks: grad, significance_level: 0.1}
            - ica_raw: {picks: meg, n_components: 64}
            - ica_autoreject: {picks: mag, ecgmethod: correlation, eogthreshold: auto}
            - interpolate_bads: {}
        """

        inputs = sorted(glob(f"{raw_dir}/Cambridge/derivatives/*/meg/*.fif"))
        outdir = f"{preproc_dir}/Cambridge"

        preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=outdir,
            overwrite=True,
            dask_client=True,
        )

    if do_nottingham:
        config = """
            preproc:
            - set_channel_types: {EEG057: eog, EEG058: eog, EEG059: ecg}
            - pick: {picks: [mag, eog, ecg]}
            - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
            - notch_filter: {freqs: 50 100, notch_widths: 2}
            - resample: {sfreq: 250}
            - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
            - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
            - bad_channels: {picks: mag, significance_level: 0.1}
            - ica_raw: {picks: mag, n_components: 64}
            - ica_autoreject: {picks: mag, ecgmethod: correlation, eogthreshold: auto}
            - interpolate_bads: {}
        """

        inputs = sorted(glob(f"{raw_dir}/Nottingham/*/meg/*.ds"))
        outdir = f"{preproc_dir}/Nottingham"

        preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=outdir,
            overwrite=True,
            dask_client=True,
        )

    if do_cardiff:
        config = """
            preproc:
            - set_channel_types: {EEG057: eog, EEG058: eog, EEG059: ecg}
            - pick: {picks: [mag, eog, ecg]}
            - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
            - notch_filter: {freqs: 50 100, notch_widths: 2}
            - resample: {sfreq: 250}
            - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
            - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
            - bad_channels: {picks: mag, significance_level: 0.1}
            - ica_raw: {picks: mag, n_components: 64}
            - ica_autoreject: {picks: mag, ecgmethod: correlation, eogthreshold: auto}
            - interpolate_bads: {}
        """

        inputs = sorted(glob(f"{raw_dir}/Cardiff/*/meg/*.ds"))
        outdir = f"{preproc_dir}/Cardiff"

        preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=outdir,
            overwrite=True,
            dask_client=True,
        )

    if do_glasgow:
        config = """
            preproc:
            - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
            - notch_filter: {freqs: 50 100, notch_widths: 2}
            - resample: {sfreq: 250}
            - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
            - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
            - bad_channels: {picks: mag, significance_level: 0.1}
            - interpolate_bads: {}
        """

        inputs = sorted(glob(f"{raw_dir}/Glasgow/*/meg/*/c,rfDC"))
        outdir = f"{preproc_dir}/Glasgow"

        preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=outdir,
            overwrite=True,
            dask_client=True,
        )
