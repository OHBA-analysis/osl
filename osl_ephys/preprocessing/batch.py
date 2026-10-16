#!/usr/bin/env python

"""Tools for batch preprocessing."""

import argparse
import warnings
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import pprint
import traceback
import re
import logging
import pickle
from pathlib import Path
from copy import deepcopy
from functools import partial, wraps
from time import localtime, strftime
from datetime import datetime
import inspect

import mne
import numpy as np
import yaml

from . import mne_wrappers, osl_wrappers
from ..utils import find_run_id, validate_outdir, process_file_inputs
from ..utils import logger as osl_logger
from ..utils.version_utils import check_version
from ..utils.misc import set_random_seed

logger = logging.getLogger(__name__)


# Decorators


def _confirm(question):
    """Ask a yes/no question on the console; returns True for yes."""
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


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
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        Data as an MNE Raw object.
    """
    if not isinstance(infile, str):
        raise ValueError(
            "infile must be a str. Got type(infile)={0}.".format(type(infile))
        )
    if " " in infile:
        raise ValueError("filename cannot contain spaces.")

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

    # EDF file
    elif os.path.splitext(infile)[1].lower() == ".edf":
        logger.info("Detected edf file format, using: mne.io.read_raw_edf")
        raw = mne.io.read_raw_edf(infile, preload=preload)

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
        logger.info("Detected EEGLAB file format, using: mne.io.read_raw_eeglab")
        raw = mne.io.read_raw_eeglab(infile, preload=preload)

    elif os.path.splitext(infile)[1] == ".con" or os.path.splitext(infile)[1] == ".sqd":
        logger.info("Detected Ricoh/KIT file format, using: mne.io.read_raw_kit")
        raw = mne.io.read_raw_kit(infile, preload=preload)

    elif os.path.splitext(infile)[1] == ".bdf":
        logger.info("Detected BDF file format, using: mne.io.read_raw_bdf")
        raw = mne.io.read_raw_bdf(infile, preload=preload)

    elif os.path.splitext(infile)[1] == ".mff":
        logger.info("Detected EGI file format, using mne.io.read_raw_egi")
        raw = mne.io.read_raw_egi(infile, preload=preload)

    # Curry
    elif os.path.splitext(infile)[1] in [
        ".dat",
        ".dap",
        ".rs3",
        ".cdt",
        ".cdt.dpa",
        ".cdt.cef",
        ".cef",
    ]:
        logger.info("Detected Curry file format, using mne.io.read_raw_curry")
        raw = mne.io.read_raw_curry(infile, preload=preload)

    # Other formats not accepted
    else:
        try:
            logger.info("Trying to automatically detect file type")
            raw = mne.io.read_raw(infile, preload=preload)
        except:
            msg = "Unable to determine file type of input {0}".format(infile)
            logger.error(msg)
            raise ValueError(msg)

    return raw


def find_func(method, target="raw", extra_funcs=None):
    """Find a preprocessing function.

    Function priority:

    1. User custom function

    2. MNE/osl-ephys wrapper

    3. MNE method on Raw or Epochs (specified by target)

    Parameters
    ----------
    method : str
        Function name.
    target : str
        Type of MNE object to preprocess. Can be ``'raw'``, ``'epochs'``, ``'evoked'``, ``'power'`` or ``'itc'``.
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

    # 2) MNE/osl-ephys Wrapper
    # Find osl-ephys function in local module
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

    # do some checks on the config
    for key in config:
        if config[key] == "None":
            config[key] = None

    # Initialise missing values in config
    if "meta" not in config:
        config["meta"] = {"event_codes": None}
    elif "event_codes" not in config["meta"]:
        config["meta"]["event_codes"] = None
    elif "versions" not in config["meta"]:
        config["meta"]["versions"] = None

    if "preproc" not in config and "group" not in config:
        raise KeyError(
            "Please specify preprocessing and/or group processing steps in config."
        )

    if "preproc" in config and config["preproc"] is not None:
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
                    "Preprocessing stage '{0}' should only have a single key".format(
                        stage
                    )
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
    else:
        config["preproc"] = None

    if "group" in config and config["group"] is not None:
        for stage in config["group"]:
            # Check each stage is a dictionary with a single key
            if not isinstance(stage, dict):
                raise ValueError(
                    "Group processing stage '{0}' is a {1} not a dict".format(
                        stage, type(stage)
                    )
                )

            if len(stage) != 1:
                raise ValueError(
                    "Group processing stage '{0}' should only have a single key".format(
                        stage
                    )
                )

            for key, val in stage.items():
                # internally we want options to be an empty dict (for now at least)
                if val in ["null", "None", None]:
                    stage[key] = {}
    else:
        config["group"] = None

    return config


def check_config_versions(config):
    """Get config from a preprocessed fif file.

    Parameters
    ----------
    config : dictionary or yaml string
        Preprocessing configuration to check.

    Raises
    ------
    AssertionError
        Raised if package version mismatch found in 'version_assert'
    Warning
        Raised if package version mismatch found in 'version_warn'
    """
    config = load_config(config)

    # Check for version and raise an error if mismatch found
    if "version_assert" in config["meta"]:
        for vers in config["meta"]["version_assert"]:
            check_version(vers, mode="assert")

    # Check for version and raise a warning if mismatch found
    if "version_warn" in config["meta"]:
        for vers in config["meta"]["version_warn"]:
            check_version(vers, mode="warn")


def get_config_from_fif(inst):
    """Get config from a preprocessed fif file.

    Reads the ``inst.info['description']`` field of a fif file to get the preprocessing config.

    Parameters
    ----------
    inst : :py:class:`mne.io.Raw <mne.io.Raw>`, :py:class:`mne.Epochs <mne.Epochs>`, :py:class:`mne.Evoked <mne.Evoked>`
        Preprocessed MNE object.

    Returns
    -------
    dict
        Preprocessing config.
    """
    config_list = re.findall(
        "%% config start %%(.*?)%% config end %%",
        inst.info["description"],
        flags=re.DOTALL,
    )
    config = []
    for config_text in config_list:
        config.append(load_config(config_text))

    return config


def append_preproc_info(dataset, config, extra_funcs=None):
    """Add to the config of already preprocessed data to ``inst.info['description']``.

    Parameters
    ----------
    dataset : dict
        Preprocessed dataset.
    config : dict
        Preprocessing config.

    Returns
    -------
    dict
        Dataset dict containing the preprocessed data edited in place.
    """
    from .. import __version__  # here to avoid circular import

    if dataset["raw"].info["description"] == None:
        dataset["raw"].info["description"] = ""

    preproc_info = (
        "\n\nOSL-EPHYS BATCH PROCESSING APPLIED ON "
        + f"{datetime.today().strftime('%d/%m/%Y %H:%M:%S')} \n"
        + f"VERSION: {__version__}\n"
        + f"%% config start %% \n{config} \n%% config end %%"
    )

    if extra_funcs is not None:
        preproc_info += "\n\nCUSTOM FUNCTIONS USED:\n"
        for func in extra_funcs:
            preproc_info += f"%% extra_funcs start %% \n{inspect.getsource(func)}\n%% extra_funcs end %%"

    dataset["raw"].info["description"] = (
        dataset["raw"].info["description"] + preproc_info
    )

    if dataset["epochs"] is not None:
        if dataset["epochs"].info["description"] == None:
            dataset["epochs"].info["description"] = ""
        dataset["epochs"].info["description"] = (
            dataset["epochs"].info["description"] + preproc_info
        )

    return dataset


def write_dataset(
    dataset, outbase, run_id, ftype="preproc-raw", overwrite=False, skip=None
):
    """Write preprocessed data to a file.

    Will write all keys in the dataset dict to disk with corresponding extensions.

    Parameters
    ----------
    dataset : dict
        Preprocessed dataset.
    outbase : str
        Path to directory to write to.
    run_id : str
        ID for the output file.
    ftype: str
        Extension for the fif file (default ``preproc-raw``)
    overwrite : bool
        Should we overwrite if the file already exists?
    skip : list or None
        List of keys to skip writing to disk. If None, we don't skip any keys.

    Output
    ------
    fif_outname : str
        The saved fif file name
    """

    if skip is None:
        skip = []
    else:
        [logger.info("Skip saving of dataset['{}']".format(key)) for key in skip]

    # Strip "_preproc-raw" or "_raw" from the run id
    for string in ["_preproc-raw", "_raw"]:
        if string in run_id:
            run_id = run_id.replace(string, "")

    if "raw" in skip:
        outnames = {"raw": None}
    else:
        outnames = {"raw": outbase.format(run_id=run_id, ftype=ftype, fext="fif")}
        if Path(outnames["raw"]).exists() and not overwrite:
            raise ValueError(
                "{} already exists. Please delete or do use overwrite=True.".format(
                    outnames["raw"]
                )
            )
        logger.info(f"Saving dataset['raw'] as {outnames['raw']}")
        dataset["raw"].save(outnames["raw"], overwrite=overwrite)

    if "events" in dataset and "events" not in skip and dataset["events"] is not None:
        outnames["events"] = outbase.format(run_id=run_id, ftype="events", fext="npy")
        logger.info(f"Saving dataset['events'] as {outnames['events']}")
        np.save(outnames["events"], dataset["events"])

    if (
        "event_id" in dataset
        and "event_id" not in skip
        and dataset["event_id"] is not None
    ):
        outnames["event_id"] = outbase.format(
            run_id=run_id, ftype="event-id", fext="yml"
        )
        logger.info(f"Saving dataset['event_id'] as {outnames['event_id']}")
        yaml.dump(dataset["event_id"], open(outnames["event_id"], "w"))

    if "epochs" in dataset and "epochs" not in skip and dataset["epochs"] is not None:
        outnames["epochs"] = outbase.format(run_id=run_id, ftype="epo", fext="fif")
        logger.info(f"Saving dataset['epochs'] as {outnames['epochs']}")
        dataset["epochs"].save(outnames["epochs"], overwrite=overwrite)

    if "ica" in dataset and "ica" not in skip and dataset["ica"] is not None:
        outnames["ica"] = outbase.format(run_id=run_id, ftype="ica", fext="fif")
        logger.info(f"Saving dataset['ica'] as {outnames['ica']}")
        dataset["ica"].save(outnames["ica"], overwrite=overwrite)

    if "tfr" in dataset and "tfr" not in skip and dataset["tfr"] is not None:
        outnames["tfr"] = outbase.format(run_id=run_id, ftype="tfr", fext="fif")
        logger.info(f"Saving dataset['tfr'] as {outnames['tfr']}")
        dataset["tfr"].save(outnames["tfr"], overwrite=overwrite)

    if "glm" in dataset and "glm" not in skip and dataset["glm"] is not None:
        outnames["glm"] = outbase.format(run_id=run_id, ftype="glm", fext="pkl")
        logger.info(f"Saving dataset['glm'] as {outnames['glm']}")
        dataset["glm"].save_pkl(outnames["glm"], overwrite=overwrite)

    if "fig" in dataset and "fig" not in skip and dataset["fig"] is not None:
        keys = dataset["fig"].keys()
        outnames["fig"] = {}
        for key in keys:
            outnames["fig"][key] = outbase.format(run_id=run_id, ftype=key, fext="png")
            logger.info(f"Saving dataset['fig'][{key}] as {outnames['fig'][key]}")
            dataset["fig"][key].savefig(outnames["fig"][key])

    # save remaining keys as pickle files
    for key in dataset:
        if key not in outnames and key not in skip:
            outnames[key] = outbase.format(run_id=run_id, ftype=key, fext="pkl")
            logger.info(f"Saving dataset['{key}'] as {outnames[key]}")
            if (
                (not os.path.exists(outnames[key]) or overwrite)
                and key not in skip
                and dataset[key] is not None
            ):
                with open(outnames[key], "wb") as f:
                    pickle.dump(dataset[key], f)
    return outnames


def read_dataset(fif, preload=False, ftype=None):
    """Reads ``fif``/``npy``/``yml`` files associated with a dataset.

    Parameters
    ----------
    fif : str
        Path to raw fif file (can be preprocessed).
    preload : bool
        Should we load the raw fif data?
    ftype : str
        Extension for the fif file (will be replaced for e.g. ``'_events.npy'`` or
        ``'_ica.fif'``). If ``None``, we assume the fif file is preprocessed with
        ``osl-ephys`` and has the extension ``'_preproc-raw'``. If this fails, we guess
        the extension as whatever comes after the last ``'_'``.

    Returns
    -------
    dataset : dict
        Contains keys: ``'raw'``, ``'events'``, ``'event_id'``, ``'epochs'``, ``'ica'``.
    """
    print("Loading dataset:")

    print("Reading", fif)
    raw = mne.io.read_raw_fif(fif, preload=preload)

    # Guess extension
    if ftype is None:
        logger.info("Guessing the preproc extension")
        if "preproc-raw" in fif:
            logger.info('Assuming fif file type is "preproc-raw"')
            ftype = "preproc-raw"
        else:
            if len(fif.split("_")) < 2:
                logger.error("Unable to guess the fif file extension")
            else:
                logger.info('Assuming fif file type is the last "_" separated string')
                ftype = fif.split("_")[-1].split(".")[-2]

    # add extension to fif file name
    ftype = ftype + ".fif"

    events = Path(fif.replace(ftype, "events.npy"))
    if events.exists():
        print("Reading", events)
        events = np.load(events)
    else:
        events = None

    event_id = Path(fif.replace(ftype, "event-id.yml"))
    if event_id.exists():
        print("Reading", event_id)
        with open(event_id, "r") as file:
            event_id = yaml.load(file, Loader=yaml.Loader)
    else:
        event_id = None

    epochs = Path(fif.replace(ftype, "epo.fif"))
    if epochs.exists():
        print("Reading", epochs)
        epochs = mne.read_epochs(epochs)
    else:
        epochs = None

    ica = Path(fif.replace(ftype, "ica.fif"))
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
    show=False,
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
    ax : :py:class:`matplotlib.axes <matplotlib.axes>`
        Matplotlib axes to plot on.
    title : str
        Title for the plot.

    Returns
    -------
    fig : :py:class:`matplotlib.figure <matplotlib.figure>`
    ax : :py:class:`matplotlib.axes <matplotlib.axes>`
    """
    config = load_config(config)

    if np.logical_or(ax == None, fig == None):
        fig = plt.figure(figsize=(8, 12))
        plt.subplots_adjust(top=0.95, bottom=0.05)
        ax = plt.subplot(111, frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    if title == None:
        ax.set_title("osl-ephys Processing Recipe", fontsize=24)
    else:
        ax.set_title(title, fontsize=24)

    tmp_h = 1
    if config["preproc"] is not None:
        tmp_h += 1 + len(config["preproc"])
    if config["group"] is not None:
        tmp_h += 1 + len(config["group"])
    stage_height = 1 / tmp_h

    box = dict(boxstyle="round", facecolor=stagecol, alpha=1, pad=0.3)
    startbox = dict(boxstyle="round", facecolor=startcol, alpha=1)
    font = {
        "family": "serif",
        "color": "k",
        "weight": "normal",
        "size": 16,
    }
    stages = [{"input": ""}]
    if config["preproc"] is not None:
        stages += [{"preproc": ""}, *config["preproc"]]
    if config["group"] is not None:
        stages += [{"group": ""}, *config["group"]]
    stages.append({"output": ""})
    stage_str = "$\\bf{{{0}}}$ {1}"

    ax.arrow(
        0.5,
        1,
        0.0,
        -1 + 0.02,
        fc="k",
        ec="k",
        head_width=0.045,
        head_length=0.035,
        length_includes_head=True,
    )

    for idx, stage in enumerate(stages):
        method, userargs = next(iter(stage.items()))

        method = method.replace("_", r"\_")
        if method in ["input", "preproc", "group", "output"]:
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


def run_proc_chain(
    config,
    infile,
    subject=None,
    ftype="preproc-raw",
    outdir=None,
    logsdir=None,
    reportdir=None,
    ret_dataset=True,
    gen_report=None,
    overwrite=False,
    skip_save=None,
    extra_funcs=None,
    random_seed="auto",
    verbose="INFO",
    mneverbose="WARNING",
):
    """Run preprocessing for a single file (v2 entry point).

    Use :func:`osl_ephys.pipeline.run_chain`. This wrapper builds the
    equivalent call; ``ftype`` and ``skip_save`` are ignored, and the output
    goes to ``outdir/preproc/<id>_preproc-raw.fif``.

    Returns
    -------
    dict or bool
        The dataset dict (empty if preprocessing failed) if
        ``ret_dataset=True``, otherwise whether preprocessing succeeded.
    """
    from osl_ephys.pipeline import run_chain

    warnings.warn(
        "run_proc_chain is replaced by osl_ephys.pipeline.run_chain.", FutureWarning
    )
    session = {
        "id": subject
        or find_run_id(str(infile) if isinstance(infile, str) else infile.filenames[0])
    }
    session["raw_file" if isinstance(infile, str) else "raw"] = infile
    if gen_report is None:
        gen_report = outdir is not None or reportdir is not None
    qc_dir = (
        (
            reportdir
            or (Path(outdir) / "qc" if outdir else Path.cwd() / "preproc_report")
        )
        if gen_report
        else None
    )
    try:
        dataset = run_chain(
            config,
            session,
            outdir=outdir,
            qc_dir=qc_dir,
            log_dir=logsdir,
            extra_funcs=extra_funcs,
            overwrite=overwrite,
            random_seed=random_seed,
            verbose=verbose,
            mneverbose=mneverbose,
        )
    except Exception:
        logger.exception("Preprocessing failed")
        return {} if ret_dataset else False
    return dataset if ret_dataset else True


def run_proc_batch(
    config,
    files,
    subjects=None,
    ftype="preproc-raw",
    outdir=None,
    logsdir=None,
    reportdir=None,
    gen_report=True,
    overwrite=False,
    skip_save=None,
    extra_funcs=None,
    covs=None,
    random_seed="auto",
    verbose="INFO",
    mneverbose="WARNING",
    strictrun=False,
    dask_client=False,
):
    """Run preprocessing for a set of files (v2 entry point).

    Use :func:`osl_ephys.pipeline.run_batch`, which runs sessions in
    parallel with ``n_workers``. This wrapper builds the equivalent call;
    ``ftype``, ``skip_save`` and ``strictrun`` are ignored.

    Returns
    -------
    flags : np.ndarray of bool
        Whether each file was processed.
    """
    from osl_ephys.pipeline import manifest, run_batch

    warnings.warn(
        "run_proc_batch is replaced by osl_ephys.pipeline.run_batch.", FutureWarning
    )
    if dask_client:
        raise ValueError(
            "dask is no longer used; pass n_workers to osl_ephys.pipeline.run_batch instead."
        )
    infiles, _, _ = process_file_inputs(files)
    sessions = manifest.from_v2(infiles, subjects)
    results = run_batch(
        config,
        sessions,
        outdir or os.getcwd(),
        n_workers=1,
        log_dir=logsdir,
        qc_dir=reportdir if gen_report else False,
        extra_funcs=extra_funcs,
        covs=covs,
        overwrite=overwrite,
        random_seed=random_seed,
        verbose=verbose,
        mneverbose=mneverbose,
    )
    return np.array([status != "failed" for _, status, _, _ in results])


def main(argv=None):
    """The ``osl_preproc`` command (v2); use ``osl-ephys run``."""
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Batch preprocess some fif files (use 'osl-ephys run')."
    )
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
        help="Overwrite previous output files",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="INFO",
        help="Logging level for osl-ephys functions",
    )
    parser.add_argument(
        "--mneverbose",
        type=str,
        default="WARNING",
        help="Logging level for MNE functions",
    )
    parser.add_argument("--strictrun", action="store_true", help="Ignored")
    args = parser.parse_args(argv)
    run_proc_batch(**vars(args))


if __name__ == "__main__":
    main()
