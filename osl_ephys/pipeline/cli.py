"""The ``osl-ephys`` command.

Subcommands: ``run`` (a config over a manifest), ``download-data`` (fetch
every data file up front) and ``check`` (test the installation).
"""

from __future__ import annotations

import argparse
import sys


def _run(args):
    from osl_ephys.pipeline import run_batch

    results = run_batch(
        args.config,
        args.sessions,
        args.outdir,
        n_workers=args.n_workers,
        log_dir=args.log_dir,
        qc_dir=False if args.no_report else args.qc_dir,
        overwrite=args.overwrite,
        verbose=args.verbose,
        mneverbose=args.mneverbose,
        surface_extraction_method=args.surface_extraction_method,
    )
    return 1 if any(status == "failed" for _, status, _, _ in results) else 0


def _download_data(args):
    from osl_ephys.files._fetch import cache_directory, fetch_all

    print(f"Downloading the osl-ephys data files to {cache_directory()}")
    path = fetch_all()
    print(f"Done. Set OSL_DATA to {path} to use this cache elsewhere.")
    return 0


def _check(args):
    """Import every subpackage, show versions, fetch a test file, run a chain."""
    from importlib.metadata import version

    import mne

    import osl_ephys
    from osl_ephys import (
        files,
        glm,
        maxfilter,
        preprocessing,
        report,
        source_recon,
    )
    from osl_ephys.pipeline import run_chain

    print(
        f"osl-ephys {osl_ephys.__version__}, mne {mne.__version__}, python {sys.version.split()[0]}"
    )
    for name in ("numpy", "scipy", "sails", "glmtools", "fslpy", "pyvista"):
        try:
            print(f"  {name} {version(name)}")
        except Exception:
            print(f"  {name} missing")
    print(f"data cache: {files._fetch.cache_directory()}")
    raw_file = files.test_data.elekta
    print(f"test recording: {raw_file}")
    dataset = run_chain(
        {"preproc": [{"resample": {"sfreq": 100}}]}, raw_file, verbose="WARNING"
    )
    print(
        f"preprocessing chain ran: {dataset['raw'].n_times} samples at {dataset['raw'].info['sfreq']} Hz"
    )
    print("osl-ephys is working.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="osl-ephys", description="OHBA Software Library for M/EEG analysis."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a config on a set of sessions")
    run.add_argument("config", help="YAML config file")
    run.add_argument(
        "sessions",
        help="CSV file with one row per session (columns: id, subject, raw_file, smri_file, ...)",
    )
    run.add_argument("outdir", help="output directory")
    run.add_argument(
        "--n-workers", type=int, default=1, help="sessions processed at the same time"
    )
    run.add_argument(
        "--log-dir",
        default=None,
        help="directory for the log files (default: outdir/logs)",
    )
    run.add_argument(
        "--qc-dir",
        default=None,
        help="directory for the QC report (default: outdir/qc)",
    )
    run.add_argument(
        "--no-report", action="store_true", help="do not generate the QC report"
    )
    run.add_argument(
        "--overwrite", action="store_true", help="rerun sessions whose output exists"
    )
    run.add_argument("--verbose", default="INFO", help="logging level of osl-ephys")
    run.add_argument("--mneverbose", default="WARNING", help="logging level of MNE")
    run.add_argument(
        "--surface-extraction-method", default="fsl", choices=["fsl", "freesurfer"]
    )
    run.set_defaults(func=_run)

    sub.add_parser(
        "download-data", help="download every data file, for use offline"
    ).set_defaults(func=_download_data)
    sub.add_parser("check", help="check the installation works").set_defaults(
        func=_check
    )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
