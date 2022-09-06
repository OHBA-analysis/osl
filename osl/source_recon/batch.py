#!/usr/bin/env python

"""Batch processing for source reconstruction.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import sys
import pathlib
import traceback
from time import localtime, strftime
from functools import partial

import numpy as np
import mne
from mne.beamformer import apply_lcmv_raw

from . import rhino, rhino_utils, beamforming, parcellation
from ..report import raw_report
from ..preprocessing import import_data
from ..utils import logger as osl_logger
from ..utils import validate_outdir, find_run_id, parallel

import logging
logger = logging.getLogger(__name__)


def run_coreg_chain(
    subject,
    preproc_file,
    smri_file,
    coreg_dir,
    model,
    pos_file=None,
    use_headshape=True,
    include_nose=False,
    use_nose=False,
    edit_polhemus_func=None,
    cleanup_files=True,
    logsdir=None,
    reportdir=None,
    gen_report=True,
    verbose="INFO",
    mneverbose="WARNING",
):
    # Make sure we have FSL installed
    rhino_utils.check_fsl()

    # Get run ID
    run_id = find_run_id(preproc_file)

    # Generate log filename
    name_base = "{run_id}_{ftype}.{fext}"
    logbase = os.path.join(logsdir, name_base)
    logfile = logbase.format(run_id=run_id, ftype="coreg", fext="log")
    mne.utils._logging.set_log_file(logfile)

    # Finish setting up loggers
    osl_logger.set_up(prefix=run_id, log_file=logfile, level=verbose, startup=False)
    mne.set_log_level(mneverbose)
    logger = logging.getLogger(__name__)
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    logger.info("{0} : Starting OSL Processing".format(now))
    logger.info("input : {0}".format(coreg_dir / subject))

    # Create directory for coregistration and report
    os.makedirs(coreg_dir / subject, exist_ok=True)
    if gen_report:
        reportdir = validate_outdir(reportdir / run_id)

    # MAIN BLOCK - Run the coregistration and catch any exceptions
    try:
        # Setup polhemus files
        current_status = "Setting up polhemus files"
        logger.info(current_status)
        if pos_file is None:
            # Use the preprocessed fif file to get positions
            (
                polhemus_headshape_file,
                polhemus_nasion_file,
                polhemus_rpa_file,
                polhemus_lpa_file,
            ) = rhino.extract_polhemus_from_info(
                fif_file=preproc_file, outdir=coreg_dir / subject
            )
        else:
            # Use the pos file
            (
                polhemus_headshape_file,
                polhemus_nasion_file,
                polhemus_rpa_file,
                polhemus_lpa_file,
            ) = rhino.extract_polhemus_from_pos_file(
                pos_file=pos_file, outdir=coreg_dir / subject
            )

        if edit_polhemus_func is not None:
            edit_polhemus_func(
                subject=subject,
                polhemus_headshape_file=polhemus_headshape_file,
                polhemus_nasion_file=polhemus_nasion_file,
                polhemus_rpa_file=polhemus_rpa_file,
                polhemus_lpa_file=polhemus_lpa_file,
            )

        # Compute surface
        current_status = "Computing surface"
        logger.info(current_status)
        rhino.compute_surfaces(
            smri_file=smri_file,
            subjects_dir=coreg_dir,
            subject=subject,
            include_nose=include_nose,
            cleanup_files=cleanup_files,
        )

        # Run coregistration
        current_state = "Coregistering"
        logger.info(current_status)
        rhino.coreg(
            fif_file=preproc_file,
            subjects_dir=coreg_dir,
            subject=subject,
            polhemus_headshape_file=polhemus_headshape_file,
            polhemus_nasion_file=polhemus_nasion_file,
            polhemus_rpa_file=polhemus_rpa_file,
            polhemus_lpa_file=polhemus_lpa_file,
            use_headshape=use_headshape,
            use_nose=use_nose,
        )

        # Compute forward model
        rhino.forward_model(subjects_dir=coreg_dir, subject=subject, model=model)

    except Exception as e:
        logger.critical("**************************")
        logger.critical("* COREGISTRATION FAILED! *")
        logger.critical("**************************")

        ex_type, ex_value, ex_traceback = sys.exc_info()
        logger.error(current_status)
        logger.error(ex_type)
        logger.error(ex_value)
        logger.error(traceback.print_tb(ex_traceback))

        with open(logfile.replace(".log", ".error.log"), "w") as f:
            f.write('Processing filed during stage : "{0}"'.format(current_status))
            f.write(str(ex_type))
            f.write("\n")
            f.write(str(ex_value))
            f.write("\n")
            traceback.print_tb(ex_traceback, file=f)

        return False

    if gen_report:
        # Save coregistration plot
        rhino.coreg_display(
            subjects_dir=coreg_dir,
            subject=subject,
            filename=reportdir / "coreg.html"
        )

        # Generate HTML data for the report
        preproc_data = import_data(preproc_file)
        raw_report.gen_html_data(
            preproc_data, reportdir, coreg=run_id + "/coreg.html"
        )

    return True


def run_coreg_batch(
    coreg_dir,
    subjects,
    preproc_files,
    smri_files,
    model,
    pos_files=None,
    use_headshape=True,
    include_nose=False,
    use_nose=False,
    edit_polhemus_func=None,
    cleanup_files=True,
    logsdir=None,
    reportdir=None,
    gen_report=True,
    verbose="INFO",
    mneverbose="WARNING",
    dask_client=False,
):
    """Batch coregistration.

    This function does the following:
    1) Sets up the Polhemus files.
    2) Computes the surface.
    3) Runs coregistration.
    4) Computes the forward model.

    Parameters
    ----------
    coreg_dir : string
        Coregistration directory.
    subjects : list of strings
        Subject names.
    preproc_files : list of strings
        Preprocessed fif files.
    smri_files : list of strings
        Structural MRI files.
    model : string
        Forward model to use.
    pos_files : list of strings
        Pos files.
    use_headshape : bool
        Should we use the headshape points?
    include_nose : bool
        Should we include the nose?
    use_nose : bool
        Should we use the nose?
    edit_polhemus_func : function
        Function to edit the headshape/nasion/rpa/lpa points.
        This function will be called with the following keyword arguments:
        edit_polhemus_func(
            subject,
            polhemus_headshape_file=polhemus_headshape_file,
            polhemus_nasion_file=polhemus_naison,
            polhemus_rpa_file=polhemus_rpa_file,
            polhemus_lpa_file=polhemus_lpa_file,
        )
    cleanup_files : bool
        Should we clean up the files?
    logsdir : string
        Path to logs directory.
    reportdir : string
        Path to report directory.
    verbose : string
        Level of verbose.
    mneverbose : string
        Level of MNE verbose.
    dask_client : bool
        Are we using a dask client?

    Returns
    -------
    flags : list of bool
        Flags indicating whether coregistration was successful.
    """
    os.makedirs(coreg_dir, exist_ok=True)
    coreg_dir = pathlib.Path(coreg_dir)
    logsdir = validate_outdir(logsdir or coreg_dir / "logs")
    if gen_report:
        # Create root report directory
        reportdir = validate_outdir(reportdir or pathlib.Path(coreg_dir) / "report")

    # Initialise Loggers
    mne.set_log_level(mneverbose)
    logfile = os.path.join(logsdir, 'osl_batch.log')
    osl_logger.set_up(log_file=logfile, level=verbose, startup=False)
    logger.info('Starting OSL Batch Coregistration')

    # Create partial function with fixed options
    pool_func = partial(
        run_coreg_chain,
        use_headshape=use_headshape,
        include_nose=include_nose,
        use_nose=use_nose,
        edit_polhemus_func=edit_polhemus_func,
        cleanup_files=cleanup_files,
        logsdir=logsdir,
        reportdir=reportdir,
        gen_report=gen_report,
        verbose=verbose,
        mneverbose=mneverbose,
    )

    if pos_files is None:
        pos_files = [None] * len(subjects)

    # Loop through input files to generate arguments for run_coreg_chain
    args = []
    for subject, preproc_file, smri_file, pos_file in zip(
        subjects, preproc_files, smri_files, pos_files
    ):
        args.append(
            (subject, preproc_file, smri_file, coreg_dir, model, pos_file)
        )

    # Actually run the processes
    if dask_client:
        flags = parallel.dask_parallel_bag(pool_func, args)
    else:
        flags = [pool_func(*aa) for aa in args]

    logger.info("Processed {0}/{1} files successfully".format(np.sum(flags), len(flags)))

    if gen_report:
        # Generate HTML report
        raw_report.gen_html_page(reportdir)

        logger.info("******************************" + "*" * len(str(reportdir)))
        logger.info(f"* REMEMBER TO CHECK REPORT: {reportdir} *")
        logger.info("******************************" + "*" * len(str(reportdir)))

    return flags


def run_bf_parc_chain(
    subject,
    preproc_file,
    chantypes,
    rank,
    freq_range,
    parcellation_file,
    orthogonalise,
    coreg_dir,
    src_dir,
    logsdir,
    verbose,
    mneverbose,
):
    # Make sure we have FSL installed
    rhino_utils.check_fsl()

    # Get run ID
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
    logger.info("input : {0}".format(coreg_dir / subject))

    try:
        # Load preprocessed data
        preproc_data = import_data(preproc_file)
        preproc_data.pick(chantypes)

        if freq_range is not None:
            # Bandpass filter
            current_status = "bandpass filtering: {}-{} Hz".format(
                freq_range[0], freq_range[1]
            )
            logger.info(current_status)
            preproc_data = preproc_data.filter(
                l_freq=freq_range[0],
                h_freq=freq_range[1],
                method="iir",
                iir_params={"order": 5, "ftype": "butter"},
            )

        # Beamforming
        current_status = "beamforming.make_lcmv"
        logger.info(current_status)
        filters, variances = beamforming.make_lcmv(
            subjects_dir=coreg_dir,
            subject=subject,
            dat=preproc_data,
            chantypes=chantypes,
            weight_norm="nai",
            rank=rank,
            batch_mode=True,
        )
        for k, v in variances.items():
            logger.info("variance for chan type {} is {}".format(k, v))

        current_status = "mne.beamforming.apply_lcmv"
        logger.info(current_status)
        src_data = apply_lcmv_raw(preproc_data, filters)
        src_ts_mni, _, src_coords_mni, _ = beamforming.transform_recon_timeseries(
            subjects_dir=coreg_dir,
            subject=subject,
            recon_timeseries=src_data.data,
            batch_mode=True,
        )

        # Parcellation
        current_status = "parcellation"
        logger.info(current_status)
        p = parcellation.Parcellation(parcellation_file)

        gridstep = int(rhino_utils.get_gridstep(src_coords_mni.T) / 1000)
        logger.info("gridstep = {} mm".format(gridstep))

        p.parcellate(
            voxel_timeseries=src_ts_mni,
            voxel_coords=src_coords_mni,
            method="spatial_basis",
            batch_mode=True,
        )
        parcel_ts = p.parcel_timeseries["data"]

        # Orthogonalisation
        if orthogonalise:
            current_status = "orthogonalisation (symmetric)"
            logger.info(current_status)
            parcel_ts = parcellation.symmetric_orthogonalise(
                parcel_ts, maintain_magnitudes=True
            )

        # Save parcellated data
        parc_data_file = src_dir / f"{subject}.npy"
        logger.info(f"Saving {parc_data_file}")
        np.save(parc_data_file, parcel_ts)

    except Exception as e:
        logger.critical("*********************************")
        logger.critical("* SOURCE RECONSTRUCTION FAILED! *")
        logger.critical("*********************************")

        ex_type, ex_value, ex_traceback = sys.exc_info()
        logger.error(current_status)
        logger.error(ex_type)
        logger.error(ex_value)
        logger.error(traceback.print_tb(ex_traceback))

        with open(logfile.replace(".log", ".error.log"), "w") as f:
            f.write('Processing filed during stage : "{0}"'.format(current_status))
            f.write(str(ex_type))
            f.write("\n")
            f.write(str(ex_value))
            f.write("\n")
            traceback.print_tb(ex_traceback, file=f)

        return False

    return True

def run_bf_parc_batch(
    preproc_files,
    subjects,
    chantypes,
    rank,
    parcellation_file,
    src_dir,
    coreg_dir,
    orthogonalise=True,
    freq_range=None,
    logsdir=None,
    verbose="INFO",
    mneverbose="WARNING",
    dask_client=False,
):
    """Batch processing for source reconstruction: beamforming, parcellation
    and orthogonalisation.

    """
    src_dir = pathlib.Path(src_dir)
    coreg_dir = pathlib.Path(coreg_dir)

    # Validation
    if freq_range is not None:
        msg = "freq_range must be a list of length 2: [l_freq, h_freq]"
        if not isinstance(freq_range, list):
            raise ValueError(msg)
        if len(freq_range) != 2:
            raise ValueError(msg)

    logsdir = validate_outdir(logsdir or src_dir / "logs")

    # Initialise Loggers
    mne.set_log_level(mneverbose)
    logfile = os.path.join(logsdir, 'osl_batch.log')
    osl_logger.set_up(log_file=logfile, level=verbose, startup=False)
    logger.info('Starting OSL Batch Source Reconstruction')

    # Create partial function with fixed options
    pool_func = partial(
        run_bf_parc_chain,
        chantypes=chantypes,
        rank=rank,
        freq_range=freq_range,
        parcellation_file=parcellation_file,
        orthogonalise=orthogonalise,
        coreg_dir=coreg_dir,
        src_dir=src_dir,
        logsdir=logsdir,
        verbose=verbose,
        mneverbose=mneverbose,
    )

    # Loop through input files to generate arguments for run_bf_parc_chain
    args = []
    for subject, preproc_file in zip(subjects, preproc_files):
        args.append((subject, preproc_file))

    # Actually run the processes
    if dask_client:
        flags = parallel.dask_parallel_bag(pool_func, args)
    else:
        flags = [pool_func(*aa) for aa in args]

    logger.info("Processed {0}/{1} files successfully".format(np.sum(flags), len(flags)))

    return flags
