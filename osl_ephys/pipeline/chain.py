"""Run a config on one session."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import mne

from osl_ephys.pipeline import config as cfg
from osl_ephys.pipeline import manifest, registry
from osl_ephys.utils import logger as osl_logger
from osl_ephys.io.file_handling import find_run_id
from osl_ephys.utils.misc import set_random_seed

logger = logging.getLogger("osl_ephys")

PREPROC_FTYPE = "preproc-raw"


def run_chain(
    config,
    session,
    outdir=None,
    qc_dir=None,
    log_dir=None,
    extra_funcs=None,
    covs=None,
    overwrite=False,
    random_seed="auto",
    verbose="INFO",
    mneverbose="WARNING",
    surface_extraction_method="fsl",
):
    """Run the per-session sections of a config on one session.

    Parameters
    ----------
    config : str or dict
        The config (see :mod:`osl_ephys.pipeline.config`).
    session : dict, str or mne.io.Raw
        A session from the manifest, the path of a raw file, or a Raw
        object already in memory.
    outdir : str, optional
        Output directory. Without it nothing is written: the preprocessing
        section runs in memory and the sections after it cannot run.
    qc_dir : str, optional
        Directory for the QC report; no report is made without it.
    log_dir : str, optional
        Directory for the log files of the source-reconstruction steps.
    extra_funcs : list of callable, optional
        Custom steps.
    covs : dict or pandas.DataFrame, optional
        Covariates for the GLM steps.
    overwrite : bool, optional
        Rerun the preprocessing when its output exists.
    random_seed : 'auto', int or None, optional
        Seed for Python and NumPy; 'auto' draws one.
    verbose, mneverbose : str, optional
        Logging levels of osl-ephys and MNE.
    surface_extraction_method : {'fsl', 'freesurfer'}, optional
        Passed to the source-reconstruction steps.

    Returns
    -------
    dataset : dict
        What the preprocessing produced (``raw``, ``events``, ``epochs``,
        ``event_id``, ``ica``, ``fig``) plus ``session`` and ``paths``.
    """
    config = cfg.load_config(config)
    cfg.validate(config, extra_funcs)
    session = _as_session(session)
    paths = manifest.output_paths(session, outdir) if outdir is not None else {}

    # Inside a worker the session logger already captures the package logger;
    # setting it up again would point its handlers at the previous session
    if getattr(osl_logger.osl_logger, "already_setup", False):
        osl_logger.set_level(verbose)
    else:
        osl_logger.set_up(prefix=session["id"], level=verbose, startup=False)
    mne.set_log_level(mneverbose)
    if random_seed == "auto":
        set_random_seed()
    elif random_seed is not None:
        set_random_seed(random_seed)

    dataset = {
        "raw": session.pop("raw", None),
        "events": None,
        "epochs": None,
        "event_id": config["meta"]["event_codes"],
        "ica": None,
        "fig": {},
        "covs": covs,
        "session": session,
        "paths": paths,
        "qc_dir": qc_dir,
        "log_dir": log_dir,
        "surface_extraction_method": surface_extraction_method,
    }

    if cfg.steps(config, "preproc"):
        dataset = _run_preproc(config, dataset, qc_dir, extra_funcs, overwrite)

    later = [
        s for s in ("coreg", "source_recon", "parcellation") if cfg.steps(config, s)
    ]
    if later:
        if outdir is None:
            raise ValueError(
                f"Sections {later} write their output to outdir, which was not given."
            )
        _run_source_recon(
            config,
            later,
            dataset,
            outdir,
            qc_dir,
            log_dir,
            extra_funcs,
            surface_extraction_method,
        )

    return dataset


def _as_session(session) -> dict:
    """A manifest row from whatever the caller passed."""
    if isinstance(session, (str, Path)):
        return {"id": find_run_id(str(session)), "raw_file": str(session)}
    if isinstance(session, mne.io.BaseRaw):
        name = (
            session.filenames[0]
            if session.filenames and session.filenames[0]
            else "raw"
        )
        return {"id": find_run_id(str(name)), "raw": session}
    if "id" not in session:
        raise ValueError(f"The session has no 'id': {session}")
    return dict(session)


def _run_preproc(config, dataset, qc_dir, extra_funcs, overwrite):
    """The preprocessing section: load, run the steps, write, report."""
    from osl_ephys.preprocessing.batch import (
        append_preproc_info,
        import_data,
        write_dataset,
    )

    session, paths = dataset["session"], dataset["paths"]
    out_file = paths.get("preproc_file")
    if out_file is not None and out_file.exists() and not overwrite:
        logger.info(
            f"Skipping preprocessing: {out_file} exists (pass overwrite=True to rerun)."
        )
        return dataset

    if dataset["raw"] is None:
        if "raw_file" not in session:
            raise ValueError(
                f"Session '{session['id']}' has no raw_file and no raw object."
            )
        dataset["raw"] = import_data(str(session["raw_file"]))

    for name, userargs in cfg.steps(config, "preproc"):
        userargs = deepcopy(userargs)
        step = registry.find_function(
            name, "preproc", extra_funcs, target=userargs.get("target", "raw")
        )
        logger.info(f"Running {name}")
        dataset = step.func(dataset, userargs)

    dataset = append_preproc_info(dataset, config, extra_funcs)

    if out_file is not None:
        paths["preproc_dir"].mkdir(parents=True, exist_ok=True)
        outbase = str(paths["preproc_dir"] / "{run_id}_{ftype}.{fext}")
        outnames = write_dataset(
            dataset, outbase, session["id"], ftype=PREPROC_FTYPE, overwrite=overwrite
        )
        dataset["outnames"] = outnames
        if qc_dir is not None:
            _preproc_report(dataset, qc_dir, outnames)
    return dataset


def _preproc_report(dataset, qc_dir, outnames):
    """The per-session preprocessing QC, with the v2 report until phase 5."""
    import matplotlib

    from osl_ephys.report import gen_html_data, gen_html_page

    report_dir = Path(qc_dir) / "preproc"
    data_dir = report_dir / dataset["session"]["id"]
    data_dir.mkdir(parents=True, exist_ok=True)
    backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    try:
        gen_html_data(
            dataset["raw"],
            data_dir,
            ica=dataset["ica"],
            events=dataset["events"],
            event_id=dataset["event_id"],
            preproc_fif_filename=outnames["raw"],
            run_id=dataset["session"]["id"],
            custom_figures=dataset["fig"] or None,
        )
        gen_html_page(report_dir)
    finally:
        matplotlib.use(backend)


def _run_source_recon(
    config,
    sections,
    dataset,
    outdir,
    qc_dir,
    log_dir,
    extra_funcs,
    surface_extraction_method,
):
    """The coregistration, source reconstruction and parcellation sections.

    Until phase 4 these call the v2 wrappers, which take the v2 arguments
    and write under ``outdir/<id>/``.
    """
    import inspect

    session, paths = dataset["session"], dataset["paths"]
    preproc_file = paths["preproc_file"]
    if not preproc_file.exists() and "preproc_file" not in session:
        v2_file = Path(outdir) / session["id"] / f"{session['id']}_preproc-raw.fif"
        if v2_file.exists():
            preproc_file = paths["preproc_file"] = v2_file
    ran_v2 = False
    v2_args = {
        "outdir": str(outdir),
        "subject": session["id"],
        "surface_extraction_method": surface_extraction_method,
        "preproc_file": (
            str(preproc_file) if preproc_file.exists() else session.get("preproc_file")
        ),
        "smri_file": session.get("smri_file"),
        "epoch_file": session.get("epoch_file"),
        "reportdir": str(Path(qc_dir) / "src") if qc_dir is not None else None,
        "logsdir": str(log_dir) if log_dir is not None else None,
    }
    for section in sections:
        for name, userargs in cfg.steps(config, section):
            step = registry.find_function(name, section, extra_funcs)
            logger.info(f"Running {name}")
            if step.kind == "legacy_src":
                wanted = inspect.signature(step.func).parameters
                kwargs = {k: v for k, v in v2_args.items() if k in wanted}
                step.func(**kwargs, **deepcopy(userargs))
                ran_v2 = True
            else:
                dataset = step.func(dataset, deepcopy(userargs))
    if ran_v2 and v2_args["reportdir"] is not None:
        from osl_ephys.report import src_report

        src_report.gen_html_data(
            _v2_source_recon_config(config),
            str(outdir),
            session["id"],
            v2_args["reportdir"],
            extra_funcs=extra_funcs,
        )
        src_report.gen_html_page(v2_args["reportdir"])


def _v2_source_recon_config(config):
    """The source-recon sections as the single list the v2 report reads."""
    steps = []
    for section in ("coreg", "source_recon", "parcellation"):
        steps += config.get(section) or []
    return {"source_recon": steps}
