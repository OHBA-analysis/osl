#!/usr/bin/env python

"""Tools for batch preprocessing.

"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>

import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import pprint
import traceback
import re
import logging
from pathlib import Path
from copy import deepcopy
from functools import partial, wraps
from time import localtime, strftime
from datetime import datetime

import mne
import numpy as np
import yaml

from . import mne_wrappers, osl_wrappers
from ..utils import find_run_id, validate_outdir, process_file_inputs, add_subdir
from ..utils import logger as osl_logger
from ..utils.parallel import dask_parallel_bag

logger = logging.getLogger(__name__)


# --------------------------------------------------------------
# Decorators


def print_custom_func_info(func):
    """Prints info for user-specified functions.

    Parameters
    ----------
    func : function
        Function to wrap.

    Returns
    -------
    function
        Wrapped function.
    """

    @wraps(func)
    def wrapper(dataset, userargs):
        logger.info("CUSTOM Stage - {}".format(func.__name__))
        logger.info("userargs: {}".format(str(userargs)))
        return func(dataset, userargs)

    return wrapper


# --------------------------------------------------------------
# Data importers


def import_data(infile, preload=True):
    """Imports data from a file.

    Parameters
    ----------
    infile : str
        Path to file to read. File can be bti, fif, ds, meg4 or vhdr.
    preload : bool
        Should we load the data in the file?

    Returns
    -------
    raw : mne.Raw
        Data as an MNE Raw object.
    """
    if not isinstance(infile, str):
        raise ValueError(
            "infile must be a str. Got type(infile)={0}.".format(type(infile))
        )

    logger.info("IMPORTING: {0}".format(infile))

    # BTI scan
    if os.path.split(infile)[1] == "c,rfDC":
        logger.info("Detected BTI file format, using: mne.io.read_raw_bti")
        if os.path.isfile(os.path.join(os.path.split(infile)[0], "hs_file")):
            head_shape_fname = "hs_file"
        else:
            head_shape_fname = None
        raw = mne.io.read_raw_bti(
            infile, head_shape_fname=head_shape_fname, preload=preload
        )

    # FIF file
    elif os.path.splitext(infile)[1] == ".fif":
        logger.info("Detected fif file format, using: mne.io.read_raw_fif")
        raw = mne.io.read_raw_fif(infile, preload=preload)

    # CTF data in ds directory
    elif os.path.splitext(infile)[1] == ".ds":
        logger.info("Detected CTF file format, using: mne.io.read_raw_ctf")
        raw = mne.io.read_raw_ctf(infile, preload=preload)
    elif os.path.splitext(infile)[1] == ".meg4":
        logger.info("Detected CTF file format, using: mne.io.read_raw_ctf")
        raw = mne.io.read_raw_ctf(os.path.dirname(infile), preload=preload)

    # Brainvision
    elif os.path.splitext(infile)[1] == ".vhdr":
        logger.info(
            "Detected brainvision file format, using: mne.io.read_raw_brainvision"
        )
        raw = mne.io.read_raw_brainvision(infile, preload=preload)

    # EEGLAB .set
    elif os.path.splitext(infile)[1] == ".set":
        logger.info(
            "Detected EEGLAB file format, using: mne.io.read_raw_eeglab"
        )
        raw = mne.io.read_raw_eeglab(infile, preload=preload)
        
    # Other formats not accepted
    else:
        msg = "Unable to determine file type of input {0}".format(infile)
        logger.error(msg)
        raise ValueError(msg)

    return raw


# --------------------------------------------------------------
# Batch processing utilities


def find_func(method, target="raw", extra_funcs=None):
    """Find a preprocessing function.

    Function priority:
    1) User custom function
    2) MNE/OSL wrapper
    3) MNE method on Raw or Epochs (specified by target)

    Parameters
    ----------
    method : str
        Function name.
    target : str
        Type of MNE object to preprocess. Can be 'raw', 'epochs', 'power' or 'itc'.
    extra_funcs : list
        List of user-defined functions.

    Returns
    -------
    function
        Function to preprocess an MNE object.
    """
    func = None

    # 1) user custom function
    if extra_funcs is not None:
        func_ind = [
            idx if (f.__name__ == method) else -1 for idx, f in enumerate(extra_funcs)
        ]
        if np.max(func_ind) > -1:
            func = extra_funcs[np.argmax(func_ind)]
            func = print_custom_func_info(func)

    # 2) MNE/OSL Wrapper
    # Find OSL function in local module
    if func is None and hasattr(osl_wrappers, "run_osl_{}".format(method)):
        func = getattr(osl_wrappers, "run_osl_{}".format(method))

    # Find MNE function in local module
    if func is None and hasattr(mne_wrappers, "run_mne_{}".format(method)):
        func = getattr(mne_wrappers, "run_mne_{}".format(method))

    # 3) MNE direct method
    if func is None:
        if target == "raw":
            if hasattr(mne.io.Raw, method) and callable(getattr(mne.io.Raw, method)):
                func = partial(mne_wrappers.run_mne_anonymous, method=method)
        elif target == "epochs":
            if hasattr(mne.Epochs, method) and callable(getattr(mne.Epochs, method)):
                func = partial(mne_wrappers.run_mne_anonymous, method=method)
        elif target in ("power", "itc"):
            if hasattr(mne.time_frequency.EpochsTFR, method) and callable(
                getattr(mne.time_frequency.EpochsTFR, method)
            ):
                func = partial(mne_wrappers.run_mne_anonymous, method=method)

    if func is None:
        logger.critical("Function not found! {}".format(method))

    return func 


def load_config(config):
    """Load config.

    Parameters
    ----------
    config : str or dict
        Path to yaml file or string to convert to dict or a dict.

    Returns
    -------
    dict
        Preprocessing config.
    """
    if type(config) not in [str, dict]:
        raise ValueError("config must be a str or dict, got {}.".format(type(config)))

    if isinstance(config, str):
        try:
            # See if we have a filepath
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            # We have a string
            config = yaml.load(config, Loader=yaml.FullLoader)

    # Initialise missing values in config
    if "meta" not in config:
        config["meta"] = {"event_codes": None}
    elif "event_codes" not in config["meta"]:
        config["meta"]["event_codes"] = None

    if "preproc" not in config:
        raise KeyError("Please specify preprocessing steps in config.")

    for stage in config["preproc"]:
        # Check each stage is a dictionary with a single key
        if not isinstance(stage, dict):
            raise ValueError(
                "Preprocessing stage '{0}' is a {1} not a dict".format(
                    stage, type(stage)
                )
            )

        if len(stage) != 1:
            raise ValueError(
                "Preprocessing stage '{0}' should only have a single key".format(stage)
            )

        for key, val in stage.items():
            # internally we want options to be an empty dict (for now at least)
            if val in ["null", "None", None]:
                stage[key] = {}

    for step in config["preproc"]:
        if config["meta"]["event_codes"] is None and "find_events" in step.values():
            raise KeyError(
                "event_codes must be passed in config if we are finding events."
            )

    return config


def get_config_from_fif(data):
    """Get config from a preprocessed fif file.

    Parameters
    ----------
    data : mne.Raw
        Preprocessing data.

    Return
    ------
    dict
        Preprocessing config.
    """
    config_list = re.findall(
        "%% config start %%(.*?)%% config end %%",
        data.info["description"],
        flags=re.DOTALL,
    )
    config = []
    for config_text in config_list:
        config.append(load_config(config_text))

    return config


def append_preproc_info(dataset, config):
    """Add to the config of already preprocessed data.

    Parameters
    ----------
    dataset : dict
        Preprocessed dataset.
    config : dict
        Preprocessing config.

    Returns
    -------
    dict
        Dataset dict containing the preprocessed data.
    """
    if dataset["raw"].info["description"] == None:
        dataset["raw"].info["description"] = ""
    preprocinfo = (
        "\n\nOSL BATCH PROCESSING APPLIED ON "
        + f"{datetime.today().strftime('%d/%m/%Y %H:%M:%S')} \n"
        + f"%% config start %% \n{config} \n%% config end %%"
    )
    dataset["raw"].info["description"] = (
        dataset["raw"].info["description"] + preprocinfo
    )

    if dataset["epochs"] is not None:
        if dataset["epochs"].info["description"] == None:
            dataset["epochs"].info["description"] = ""
        dataset["epochs"].info["description"] = (
            dataset["epochs"].info["description"] + preprocinfo
        )

    return dataset


def write_dataset(dataset, outbase, run_id, overwrite=False):
    """Write preprocessed data to a file.

    Parameters
    ----------
    dataset : dict
        Preprocessed dataset.
    outbase : str
        Path to directory to write to.
    run_id : str
        ID for the output file.
    overwrite : bool
        Should we overwrite if the file already exists?

    Output
    ------
    fif_outname : str
        The saved fif file name
    """

    if "preproc_raw" in run_id:
        fif_outname = outbase.format(
            run_id=run_id.replace("_preproc_raw", ""), ftype="preproc_raw", fext="fif"
        )
    else:
        fif_outname = outbase.format(
            run_id=run_id.replace("_raw", ""), ftype="preproc_raw", fext="fif"
        )
    if Path(fif_outname).exists() and not overwrite:
        raise ValueError(
            "{} already exists. Please delete or do use overwrite=True.".format(fif_outname)
        )
    dataset["raw"].save(fif_outname, overwrite=overwrite)

    if dataset["events"] is not None:
        outname = outbase.format(run_id=run_id, ftype="events", fext="npy")
        np.save(outname, dataset["events"])

    if dataset["event_id"] is not None:
        outname = outbase.format(run_id=run_id, ftype="event-id", fext="yml")
        yaml.dump(dataset["event_id"], open(outname, "w"))

    if dataset["epochs"] is not None:
        outname = outbase.format(run_id=run_id, ftype="epo", fext="fif")
        dataset["epochs"].save(outname, overwrite=overwrite)

    if dataset["ica"] is not None:
        outname = outbase.format(run_id=run_id, ftype="ica", fext="fif")
        dataset["ica"].save(outname, overwrite=overwrite)

    return fif_outname

def read_dataset(fif, preload=False):
    """Reads fif/npy/yml files associated with a dataset.

    Parameters
    ----------
    fif : str
        Path to raw fif file (can be preprocessed).
    preload : bool
        Should we load the raw fif data?

    Returns
    -------
    dataset : dict
        Contains keys: 'raw', 'events', 'epochs', 'ica'.
    """
    print("Loading dataset:")

    print("Reading", fif)
    raw = mne.io.read_raw_fif(fif, preload=preload)

    events = Path(fif.replace("preproc_raw.fif", "events.npy"))
    if events.exists():
        print("Reading", events)
        events = np.load(events)
    else:
        events = None

    event_id = Path(fif.replace("preproc_raw.fif", "event-id.yml"))
    if event_id.exists():
        print("Reading", event_id)
        with open(event_id, "r") as file:
            event_id = yaml.safe_load(file)
    else:
        event_id = None

    epochs = Path(fif.replace("preproc_raw", "epo"))
    if epochs.exists():
        print("Reading", epochs)
        epochs = mne.read_epochs(epochs)
    else:
        epochs = None

    ica = Path(fif.replace("preproc_raw", "ica"))
    if ica.exists():
        print("Reading", ica)
        ica = mne.preprocessing.read_ica(ica)
    else:
        ica = None

    dataset = {
        "raw": raw,
        "events": events,
        "event_id": event_id,
        "epochs": epochs,
        "ica": ica,
    }

    return dataset


def plot_preproc_flowchart(
    config,
    outname=None,
    show=True,
    stagecol="wheat",
    startcol="red",
    fig=None,
    ax=None,
    title=None,
):
    """Make a summary flowchart of a preprocessing chain.

    Parameters
    ----------
    config : dict
        Preprocessing config to plot.
    outname : str
        Output filename.
    show : bool
        Should we show the plot?
    stagecol : str
        Stage colour.
    startcol : str
        Start colour.
    fig : matplotlib.figure
        Matplotlib figure to plot on.
    ax : matplotlib.axes
        Matplotlib axes to plot on.
    title : str
        Title for the plot.

    Returns
    -------
    fig : matplotlib.figure
    ax : matplotlib.axes
    """
    config = load_config(config)

    if np.logical_or(ax == None, fig == None):
        fig = plt.figure(figsize=(8, 12))
        plt.subplots_adjust(top=0.95, bottom=0.05)
        ax = plt.subplot(111, frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    if title == None:
        ax.set_title("OSL Preprocessing Recipe", fontsize=24)
    else:
        ax.set_title(title, fontsize=24)

    stage_height = 1 / (1 + len(config["preproc"]))

    box = dict(boxstyle="round", facecolor=stagecol, alpha=1, pad=0.3)
    startbox = dict(boxstyle="round", facecolor=startcol, alpha=1)
    font = {
        "family": "serif",
        "color": "k",
        "weight": "normal",
        "size": 16,
    }

    stages = [{"input": ""}, *config["preproc"], {"output": ""}]
    stage_str = "$\\bf{{{0}}}$ {1}"

    ax.arrow(
        0.5, 1, 0.0, -1, fc="k", ec="k", head_width=0.045,
        head_length=0.035, length_includes_head=True,
    )

    for idx, stage in enumerate(stages):
        method, userargs = next(iter(stage.items()))

        method = method.replace("_", "\_")
        if method in ["input", "output"]:
            b = startbox
        else:
            b = box
            method = method + ":"

        ax.text(
            0.5,
            1 - stage_height * idx,
            stage_str.format(method, str(userargs)[1:-1]),
            ha="center",
            va="center",
            bbox=b,
            fontdict=font,
            wrap=True,
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.25, 0.75)

    if outname is not None:
        fig.savefig(outname, dpi=300, transparent=True)

    if show is True:
        fig.show()

    return fig, ax


# --------------------------------------------------------------
# Batch processing


def run_proc_chain(
    config,
    infile,
    outname=None,
    outdir=None,
    logsdir=None,
    reportdir=None,
    ret_dataset=True,
    gen_report=None,
    overwrite=False,
    extra_funcs=None,
    verbose="INFO",
    mneverbose="WARNING",
):
    """Run preprocessing for a single file.

    Parameters
    ----------
    config : str or dict
        Preprocessing config.
    infile : str
        Path to input file.
    outname : str
        Output filename.
    outdir : str
        Output directory. If processing multiple files, they can
        be put in unique sub directories by including {x:0} at 
        the end of the outdir, where x is the pattern which
        precedes the unique identifier and 0 is the length of 
        the unique identifier. For example: if the outdir is
        ../../{sub-:3} and each is like 
        /sub-001_task-rest.fif, the outdir for the file will be
        ../../sub-001/
    logsdir : str
        Directory to save log files to.
    reportdir : str
        Directory to save report files to.
    ret_dataset : bool
        Should we return a dataset dict?
    gen_report : bool
        Should we generate a report?
    overwrite : bool
        Should we overwrite the output file if it already exists?
    extra_funcs : list
        User-defined functions.
    verbose : str
        Level of info to print.
        Can be: CRITICAL, ERROR, WARNING, INFO, DEBUG or NOTSET.
    mneverbose : str
        Level of info from MNE to print.
        Can be: CRITICAL, ERROR, WARNING, INFO, DEBUG or NOTSET.

    Returns
    -------
    dict or bool
        If ret_dataset=True, a dict containing the preprocessed dataset with the
        following keys: raw, ica, epochs, events, event_id. An empty dict is returned
        if preprocessing fail. If return an empty dict. if ret_dataset=False, we
        return a flag indicating whether preprocessing was successful.
    """

    # Generate a run ID
    if outname is None:
        run_id = find_run_id(infile)
    else:
        run_id = os.path.splitext(outname)[0]
    name_base = "{run_id}_{ftype}.{fext}"

    if not ret_dataset:
        # Let's make sure we have an output directory
        outdir = outdir or os.getcwd()

    if outdir is not None:
        # We're saving the output to disk

        # Generate a report by default, this is overriden if the user passes
        # gen_report=False
        gen_report = True if gen_report is None else gen_report
        
        # Create output directories if they don't exist
        outdir = add_subdir(infile, outdir, run_id)
        outdir = validate_outdir(outdir)
        logsdir = validate_outdir(logsdir or outdir / "logs")
        reportdir = validate_outdir(reportdir or outdir / "report")

    else:
        # We're not saving the output to disk

        # Don't generate a report by default, this is overriden if the user passes
        # something for reportdir or gen_report=True
        gen_report = gen_report or reportdir is not None or False
        if gen_report:
            # Make sure we have a directory to write the report to
            reportdir = validate_outdir(reportdir or os.getcwd() + "/report")

        # Allow the user to create a log if they pass logsdir
        if logsdir is not None:
            logsdir = validate_outdir(logsdir)

    # Create output filename
    if outdir is not None:
        outbase = os.path.join(outdir, name_base)

    # Generate log filename
    if logsdir is not None:
        logbase = os.path.join(logsdir, name_base)
        logfile = logbase.format(
            run_id=run_id.replace("_raw", ""), ftype="preproc_raw", fext="log"
        )
        mne.utils._logging.set_log_file(logfile, overwrite=overwrite)
    else:
        logfile = None

    # Finish setting up loggers
    osl_logger.set_up(prefix=run_id, log_file=logfile, level=verbose, startup=False)
    mne.set_log_level(mneverbose)
    logger = logging.getLogger(__name__)
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    logger.info("{0} : Starting OSL Processing".format(now))
    logger.info("input : {0}".format(infile))

    # Write preprocessed data to output directory
    if outdir is not None:
        # Check for existing outputs - should be a .fif at least
        fifout = outbase.format(
            run_id=run_id.replace('_raw', ''), ftype='preproc_raw', fext='fif'
        )
        if os.path.exists(fifout) and (overwrite is False):
            logger.critical('Skipping preprocessing - existing output detected')
            return False

    # Load config
    if not isinstance(config, dict):
        config = load_config(config)

    # MAIN BLOCK - Run the preproc chain and catch any exceptions
    try:
        if isinstance(infile, str):
            raw = import_data(infile)
        elif (isinstance(infile, mne.io.fiff.raw.Raw)
              or isinstance(infile, mne.io.curry.curry.RawCurry)):
            raw = infile
            infile = raw.filenames[0]  # assuming only one file here

        # Create a dataset dict to hold the preprocessed dataset
        dataset = {
            "raw": raw,
            "events": None,
            "epochs": None,
            "event_id": config["meta"]["event_codes"],
            "ica": None,
        }

        # Do the preprocessing
        for stage in deepcopy(config["preproc"]):
            method, userargs = next(iter(stage.items()))
            target = userargs.get("target", "raw")  # Raw is default
            func = find_func(method, target=target, extra_funcs=extra_funcs)
            # Actual function call
            dataset = func(dataset, userargs)

        # Add preprocessing info to dataset dict
        dataset = append_preproc_info(dataset, config)

        fif_outname = None
        if outdir is not None:
            fif_outname = write_dataset(dataset, outbase, run_id, overwrite=overwrite)

        # Generate report data
        if gen_report:
            # Switch to non-GUI plotting backend
            mpl_backend = matplotlib.pyplot.get_backend()
            matplotlib.use('Agg')

            from ..report import gen_html_data, gen_html_page  # avoids circular import
            logger.info("{0} : Generating Report".format(now))
            report_data_dir = validate_outdir(reportdir / Path(fif_outname).stem)
            gen_html_data(
                dataset["raw"],
                report_data_dir,
                ica=dataset["ica"],
                preproc_fif_filename=fif_outname,
            )
            gen_html_page(reportdir)

            # Restore plotting context
            matplotlib.use(mpl_backend)

    except Exception as e:
        # Preprocessing failed

        if 'method' not in locals():
            method = 'import_data'
            func = import_data

        logger.critical("**********************")
        logger.critical("* PROCESSING FAILED! *")
        logger.critical("**********************")

        ex_type, ex_value, ex_traceback = sys.exc_info()
        logger.error("{0} : {1}".format(method, func))
        logger.error(ex_type)
        logger.error(ex_value)
        logger.error(traceback.print_tb(ex_traceback))

        with open(logfile.replace(".log", ".error.log"), "w") as f:
            f.write('Processing filed during stage : "{0}"'.format(method))
            f.write(str(ex_type))
            f.write("\n")
            f.write(str(ex_value))
            f.write("\n")
            traceback.print_tb(ex_traceback, file=f)

        if ret_dataset:
            # We return an empty dict to indicate preproc failed
            # This ensures the function consistently returns one
            # variable type
            return {}
        else:
            return False

    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    logger.info("{0} : Processing Complete".format(now))

    if fif_outname is not None:
        logger.info("Output file is {}".format(fif_outname))


    if ret_dataset:
        return dataset
    else:
        return True


def run_proc_batch(
    config,
    files,
    outdir=None,
    logsdir=None,
    reportdir=None,
    gen_report=True,
    overwrite=False,
    extra_funcs=None,
    verbose="INFO",
    mneverbose="WARNING",
    strictrun=False,
    dask_client=False,
):
    """Run batched preprocessing.

    This function will write output to disk (i.e. will not return the preprocessed
    data).

    Parameters
    ----------
    config : str or dict
        Preprocessing config.
    files : str or list or mne.Raw
        Can be a list of Raw objects or a list of filenames (or .ds dir names if CTF data)
        or a path to a textfile list of filenames (or .ds dir names if CTF data).
    outdir : str
        Output directory. If processing multiple files, they can
        be put in unique sub directories by including {x:0} at 
        the end of the outdir, where x is the pattern which
        precedes the unique identifier and 0 is the length of 
        the unique identifier. For example: if the outdir is
        ../../{sub-:3} and each is like /sub-001_task-rest.fif, 
        the outdir for the file will be ../../sub-001/
    logsdir : str
        Directory to save log files to.
    reportdir : str
        Directory to save report files to.
    gen_report : bool
        Should we generate a report?
    overwrite : bool
        Should we overwrite the output file if it exists?
    extra_funcs : list
        User-defined functions.
    verbose : str
        Level of info to print.
        Can be: CRITICAL, ERROR, WARNING, INFO, DEBUG or NOTSET.
    mneverbose : str
        Level of info from MNE to print.
        Can be: CRITICAL, ERROR, WARNING, INFO, DEBUG or NOTSET.
    strictrun : bool
        Should we ask for confirmation of user inputs before starting?
    dask_client : bool
        Indicate whether to use a previously initialised dask.distributed.Client
        instance.

    Returns
    -------
    list of bool
        Flags indicating whether preprocessing was successful for each input file.
    """

    if outdir is None:
        # Use the current working directory
        outdir = os.getcwd()

    # Validate the parent outdir - later do so for each subdirectory
    tmpoutdir = validate_outdir(outdir.split('{')[0])
    logsdir = validate_outdir(logsdir or tmpoutdir / "logs")
    reportdir = validate_outdir(reportdir or tmpoutdir / "report")

    # Initialise Loggers
    mne.set_log_level(mneverbose)
    if strictrun and verbose not in ['INFO', 'DEBUG']:
        # override logger level if strictrun requested but user won't see any info...
        verobse = 'INFO'
    logfile = os.path.join(logsdir, 'osl_batch.log')
    osl_logger.set_up(log_file=logfile, level=verbose, startup=False)

    logger.info('Starting OSL Batch Processing')

    # Check through inputs and parameters
    infiles, outnames, good_files = process_file_inputs(files)
    if strictrun and click.confirm('Is this correct set of inputs?') is False:
        logger.critical('Stopping : User indicated incorrect number of input files')
        sys.exit(1)
    else:
        if strictrun:
            logger.info('User confirms input files')

    logger.info('Outputs saving to: {0}'.format(outdir))
    if strictrun and click.confirm('Is this correct output directory?') is False:
        logger.critical('Stopping : User indicated incorrect output directory')
        sys.exit(1)
    else:
        if strictrun:
            logger.info('User confirms output directory')

    config = load_config(config)
    config_str = pprint.PrettyPrinter().pformat(config)
    logger.info('Running config\n {0}'.format(config_str))
    if strictrun and click.confirm('Is this the correct config?') is False:
        logger.critical('Stopping : User indicated incorrect preproc config')
        sys.exit(1)
    else:
        if strictrun:
            logger.info('User confirms input config')

    # Create partial function with fixed options
    pool_func = partial(
        run_proc_chain,
        outdir=outdir,
        logsdir=logsdir,
        reportdir=reportdir,
        ret_dataset=False,
        gen_report=gen_report,
        overwrite=overwrite,
        extra_funcs=extra_funcs,
    )

    # Loop through input files to generate arguments for run_proc_chain
    args = []

    for infile, outname in zip(infiles, outnames):
        args.append((config, infile, outname))

    # Actually run the processes
    if dask_client:
        proc_flags = dask_parallel_bag(pool_func, args)
    else:
        proc_flags = [pool_func(*aa) for aa in args]

    osl_logger.set_up(log_file=logfile, level=verbose, startup=False)
    logger.info(
        "Processed {0}/{1} files successfully".format(
            np.sum(proc_flags), len(proc_flags)
        )
    )

    # Generate a report
    if gen_report and len(infiles) > 0:
        from ..report import raw_report # avoids circular import
        raw_report.gen_html_page(reportdir)
        if raw_report.gen_html_summary(reportdir):
            logger.info("******************************" + "*" * len(str(reportdir)))
            logger.info(f"* REMEMBER TO CHECK REPORT: {reportdir} *")
            logger.info("******************************" + "*" * len(str(reportdir)))

    # Return flags
    return proc_flags


# ----------------------------------------------------------
# Main CLI user function


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Batch preprocess some fif files.")
    parser.add_argument("config", type=str, help="yaml defining preproc")
    parser.add_argument(
        "files",
        type=str,
        help="plain text file containing full paths to files to be processed",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Path to output directory to save data in",
    )
    parser.add_argument(
        "--logsdir", type=str, default=None, help="Path to logs directory"
    )
    parser.add_argument(
        "--reportdir", type=str, default=None, help="Path to report directory"
    )
    parser.add_argument(
        "--gen_report", type=bool, default=True, help="Should we generate a report?"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite previous output files if they're in the way",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="INFO",
        help="Set the logging level for OSL functions",
    )
    parser.add_argument(
        "--mneverbose",
        type=str,
        default="WARNING",
        help="Set the logging level for MNE functions",
    )
    parser.add_argument(
        "--strictrun",
        action="store_true",
        help="Will ask the user for confirmation before starting",
    )

    parser.usage = parser.format_help()
    args = parser.parse_args(argv)

    run_proc_batch(**vars(args))


if __name__ == "__main__":
    main()
