"""Batch processing for source reconstruction."""

import os
import warnings


import numpy as np
import yaml
import mne

from . import _v2_rhino as rhino, wrappers, freesurfer_utils
from ..report import src_report
from ..utils import logger as osl_logger
from ..utils import validate_outdir, find_run_id
from ..utils.misc import set_random_seed

import logging

logger = logging.getLogger(__name__)


def load_config(config):
    """Load config.

    Parameters
    ----------
    config : str or dict
        Path to yaml file or str to convert to dict or a dict.

    Returns
    -------
    config : dict
        Source reconstruction config.
    """
    if type(config) not in [str, dict]:
        raise ValueError("config must be a str or dict, got {}.".format(type(config)))

    if isinstance(config, str):
        try:
            # See if we have a filepath
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            # We have a str
            config = yaml.load(config, Loader=yaml.FullLoader)

    # Validation
    if "source_recon" not in config:
        raise ValueError("source_recon must be included in the config.")

    return config


def find_func(method, extra_funcs):
    """Find a source reconstruction function.

    Parameters
    ----------
    method : str
        Function name.
    extra_funcs : list of functions
        Custom functions.

    Returns
    -------
    func : function
        Function to use.
    """
    func = None

    # Look in custom functions
    if extra_funcs is not None:
        func_ind = [
            idx if (f.__name__ == method) else -1 for idx, f in enumerate(extra_funcs)
        ]
        if np.max(func_ind) > -1:
            func = extra_funcs[np.argmax(func_ind)]

    # Look in osl_ephys.source_recon.wrappers
    if func is None and hasattr(wrappers, method):
        func = getattr(wrappers, method)

    return func


def run_src_chain(
    config,
    outdir,
    subject,
    preproc_file=None,
    smri_file=None,
    epoch_file=None,
    surface_extraction_method="fsl",
    logsdir=None,
    reportdir=None,
    gen_report=True,
    verbose="INFO",
    mneverbose="WARNING",
    extra_funcs=None,
    random_seed="auto",
):
    """Source reconstruction for one subject (v2 entry point).

    Use :func:`osl_ephys.pipeline.run_chain` with a config that has
    ``coreg``, ``source_recon`` and ``parcellation`` sections.

    Returns
    -------
    flag : bool
        Whether source reconstruction succeeded.
    """
    from osl_ephys.pipeline import run_chain

    warnings.warn(
        "run_src_chain is replaced by osl_ephys.pipeline.run_chain.", FutureWarning
    )
    session = {"id": subject}
    for key, value in (
        ("preproc_file", preproc_file),
        ("smri_file", smri_file),
        ("epoch_file", epoch_file),
    ):
        if value not in (None, True, False):
            session[key] = str(value)
    try:
        run_chain(
            config,
            session,
            outdir=outdir,
            qc_dir=reportdir if gen_report else None,
            log_dir=logsdir,
            extra_funcs=extra_funcs,
            random_seed=random_seed,
            verbose=verbose,
            mneverbose=mneverbose,
            surface_extraction_method=surface_extraction_method,
        )
    except Exception:
        logger.exception("Source reconstruction failed")
        return False
    return True


def run_src_batch(
    config,
    outdir,
    subjects,
    preproc_files=None,
    smri_files=None,
    epoch_files=None,
    surface_extraction_method="fsl",
    logsdir=None,
    reportdir=None,
    gen_report=True,
    verbose="INFO",
    mneverbose="WARNING",
    extra_funcs=None,
    dask_client=False,
    random_seed="auto",
):
    """Source reconstruction for a set of subjects (v2 entry point).

    Use :func:`osl_ephys.pipeline.run_batch`, which runs sessions in
    parallel with ``n_workers``.

    Returns
    -------
    flags : list of bool
        Whether each subject was processed.
    """
    from osl_ephys.pipeline import manifest, run_batch

    warnings.warn(
        "run_src_batch is replaced by osl_ephys.pipeline.run_batch.", FutureWarning
    )
    if dask_client:
        raise ValueError(
            "dask is no longer used; pass n_workers to osl_ephys.pipeline.run_batch instead."
        )
    n = len(subjects)
    lists = []
    for values in (smri_files, preproc_files, epoch_files):
        if values in (None, True, False):
            lists.append(None)
        elif isinstance(values, str):
            lists.append([values] * n)
        else:
            lists.append(list(values))
    sessions = manifest.from_v2(None, subjects, *lists)
    results = run_batch(
        config,
        sessions,
        outdir,
        n_workers=1,
        log_dir=logsdir,
        qc_dir=reportdir if gen_report else False,
        extra_funcs=extra_funcs,
        random_seed=random_seed,
        verbose=verbose,
        mneverbose=mneverbose,
        surface_extraction_method=surface_extraction_method,
    )
    return [status != "failed" for _, status, _, _ in results]
