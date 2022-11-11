#!/usr/bin/env python

"""Batch processing for source reconstruction.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import sys
import traceback
import pprint
import inspect
from copy import deepcopy
from time import localtime, strftime
from functools import partial

import numpy as np
import yaml
import mne

from . import rhino, wrappers
from ..preprocessing import read_dataset
from ..report import src_report
from ..utils import logger as osl_logger
from ..utils import validate_outdir, find_run_id, parallel

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

    # Look in osl.source_recon.wrappers
    if func is None and hasattr(wrappers, method):
        func = getattr(wrappers, method)

    return func


def run_src_chain(
    config,
    src_dir,
    subject,
    preproc_file=None,
    smri_file=None,
    verbose="INFO",
    mneverbose="WARNING",
    extra_funcs=None,
):
    """Source reconstruction.

    Parameters
    ----------
    config : str or dict
        Source reconstruction config.
    src_dir : str
        Source reconstruction directory.
    subject : str
        Subject name.
    preproc_file : str
        Preprocessed fif file.
    smri_file : str
        Structural MRI file.
    verbose : str
        Level of verbose.
    mneverbose : str
        Level of MNE verbose.
    extra_funcs : list of functions
        Custom functions.

    Returns
    -------
    flag : bool
        Flag indicating whether source reconstruction was successful.
    """
    rhino.fsl_wrappers.check_fsl()

    # Directories
    src_dir = validate_outdir(src_dir)
    logsdir = validate_outdir(src_dir / "logs")
    reportdir = validate_outdir(src_dir / "report")

    # Get run ID
    if preproc_file is None:
        run_id = subject
    else:
        run_id = find_run_id(preproc_file)

    # Generate log filename
    name_base = "{run_id}_{ftype}.{fext}"
    logbase = os.path.join(logsdir, name_base)
    logfile = logbase.format(run_id=run_id, ftype="src", fext="log")
    mne.utils._logging.set_log_file(logfile)

    # Finish setting up loggers
    osl_logger.set_up(prefix=run_id, log_file=logfile, level=verbose, startup=False)
    mne.set_log_level(mneverbose)
    logger = logging.getLogger(__name__)
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    logger.info("{0} : Starting OSL Processing".format(now))
    logger.info("input : {0}".format(src_dir / subject))

    # Load config
    if not isinstance(config, dict):
        config = load_config(config)

    # Validation
    doing_coreg = (
        any(["coregister" in method for method in config["source_recon"]]) or
        any(["compute_surfaces" in method for method in config["source_recon"]]) or
        any(["coreg" in method for method in config["source_recon"]]) or
        any(["forward_model" in method for method in config["source_recon"]])
    )
    if doing_coreg and smri_file is None:
        raise ValueError("smri_file must be passed if we're doing coregistration.")

    # MAIN BLOCK - Run source reconstruction and catch any exceptions
    try:
        for stage in deepcopy(config["source_recon"]):
            method, userargs = next(iter(stage.items()))
            func = find_func(method, extra_funcs=extra_funcs)
            if func is None:
                avail_funcs = inspect.getmembers(wrappers, inspect.isfunction)
                avail_names = [name for name, _ in avail_funcs]
                if method not in avail_names:
                    raise NotImplementedError(
                        f"{method} not available.\n"
                        + "Please pass via extra_funcs "
                        + f"or use available functions: {avail_names}."
                    )
            func(src_dir, subject, preproc_file, smri_file, logger, **userargs)

    except Exception as e:
        logger.critical("*********************************")
        logger.critical("* SOURCE RECONSTRUCTION FAILED! *")
        logger.critical("*********************************")

        ex_type, ex_value, ex_traceback = sys.exc_info()
        logger.error("{0} : {1}".format(method, func))
        logger.error(ex_type)
        logger.error(ex_value)
        logger.error(traceback.print_tb(ex_traceback))

        with open(logfile.replace(".log", ".error.log"), "w") as f:
            f.write('Processing failed during stage : "{0}"'.format(method))
            f.write(str(ex_type))
            f.write("\n")
            f.write(str(ex_value))
            f.write("\n")
            traceback.print_tb(ex_traceback, file=f)

        return False

    # Generate HTML data for the report
    src_report.gen_html_data(config, src_dir, subject, reportdir, logger)

    return True


def run_src_batch(
    config,
    src_dir,
    subjects,
    preproc_files=None,
    smri_files=None,
    verbose="INFO",
    mneverbose="WARNING",
    extra_funcs=None,
    dask_client=False,
):
    """Batch source reconstruction.

    Parameters
    ----------
    config : str or dict
        Source reconstruction config.
    src_dir : str
        Source reconstruction directory.
    subjects : list of strs
        Subject names.
    preproc_files : list of strs
        Preprocessed fif files.
    smri_files : list of strs
        Structural MRI files.
    verbose : str
        Level of verbose.
    mneverbose : str
        Level of MNE verbose.
    extra_funcs : list of functions
        Custom functions.
    dask_client : bool
        Are we using a dask client?

    Returns
    -------
    flags : list of bool
        Flags indicating whether coregistration was successful.
    """
    rhino.fsl_wrappers.check_fsl()

    # Directories
    src_dir = validate_outdir(src_dir)
    logsdir = validate_outdir(src_dir / "logs")
    reportdir = validate_outdir(src_dir / "report")

    # Initialise Loggers
    mne.set_log_level(mneverbose)
    logfile = os.path.join(logsdir, 'osl_batch.log')
    osl_logger.set_up(log_file=logfile, level=verbose, startup=False)
    logger.info('Starting OSL Batch Source Reconstruction')

    # Load config
    config = load_config(config)
    config_str = pprint.PrettyPrinter().pformat(config)
    logger.info('Running config\n {0}'.format(config_str))

    # Validation
    n_subjects = len(subjects)
    if preproc_files != None:
        n_preproc_files = len(preproc_files)
        if n_subjects != n_preproc_files:
            raise ValueError(
                f"Got {n_subjects} subjects and {n_preproc_files} preproc_files."
            )

    doing_coreg = (
        any(["coregister" in method for method in config["source_recon"]]) or
        any(["compute_surfaces" in method for method in config["source_recon"]]) or
        any(["coreg" in method for method in config["source_recon"]]) or
        any(["forward_model" in method for method in config["source_recon"]])
    )
    if doing_coreg and smri_files is None:
        raise ValueError("smri_files must be passed if we are coregistering.")
    elif smri_files is None:
        smri_files = [None] * n_subjects

    if preproc_files is None:
        preproc_files = [None] * n_subjects

    # Create partial function with fixed options
    pool_func = partial(
        run_src_chain,
        verbose=verbose,
        mneverbose=mneverbose,
        extra_funcs=extra_funcs,
    )

    # Loop through input files to generate arguments for run_coreg_chain
    args = []
    for subject, preproc_file, smri_file, in zip(
        subjects, preproc_files, smri_files
    ):
        args.append((config, src_dir, subject, preproc_file, smri_file))

    # Actually run the processes
    if dask_client:
        flags = parallel.dask_parallel_bag(pool_func, args)
    else:
        flags = [pool_func(*aa) for aa in args]

    logger.info("Processed {0}/{1} files successfully".format(np.sum(flags), len(flags)))

    # Generate individual subject HTML report
    src_report.gen_html_page(reportdir)

    # Generate a summary report
    if src_report.gen_html_summary(reportdir):
        logger.info("******************************" + "*" * len(str(reportdir)))
        logger.info(f"* REMEMBER TO CHECK REPORT: {reportdir} *")
        logger.info("******************************" + "*" * len(str(reportdir)))

    return flags
