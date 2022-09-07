#!/usr/bin/env python

"""Batch processing for source reconstruction.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import sys
import pathlib
import traceback
import pprint
from time import localtime, strftime
from functools import partial

import numpy as np
import yaml
import mne
from mne.beamformer import apply_lcmv_raw

from . import rhino, rhino_utils, beamforming, parcellation
from ..report import raw_report
from ..preprocessing import import_data
from ..utils import logger as osl_logger
from ..utils import validate_outdir, find_run_id, parallel

import logging
logger = logging.getLogger(__name__)


def load_config(config):
    """Load config.

    Parameters
    ----------
    config : str or dict
        Path to yaml file or string to convert to dict or a dict.

    Returns
    -------
    dict
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
            # We have a string
            config = yaml.load(config, Loader=yaml.FullLoader)

    return config


def _validate_config(config):
    example_config = """Correct example:
    config = '''
    coregistration:
        model: Single Layer
        include_nose: true
        use_nose: true
        use_headshape: true
    beamforming:
        freq_range: [1, 45]
        chantypes: meg
        ranks: 60
    parcellation:
        file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
    '''"""
    config_keys = [str(c) for c in config.keys()]
    correct_keys = ["coregistration", "beamforming", "parcellation"]
    for key in config_keys:
        if key not in correct_keys:
            raise ValueError(f"{key} invalid. {example_config}")

    if "coregistration" in config_keys:
        coreg_example_config = """Correct example:
        config = '''
        coregistration:
            model: Single Layer
            include_nose: true
            use_nose: true
            use_headshape: true
        '''"""
        coreg_keys = [str(c) for c in config["coregistration"].keys()]
        correct_keys = ["model", "include_nose", "use_nose", "use_headshape"]
        for key in coreg_keys:
            if key not in correct_keys:
                raise ValueError(f"{key} invalid. {coreg_example_config}")
        for key in correct_keys:
            if key not in coreg_keys:
                raise ValueError(f"{key} missing. {coreg_example_config}")

    if "beamforming" in config_keys:
        bf_example_config = """Correct example:
        config = '''
        beamforming:
            freq_range: [1, 45]
            chantypes: meg
            ranks: 60
        '''"""
        bf_keys = [str(c) for c in config["beamforming"].keys()]
        correct_keys = ["freq_range", "chantypes", "ranks"]
        for key in bf_keys:
            if key not in correct_keys:
                raise ValueError(f"{key} invalid. {bf_example_config}")
        for key in correct_keys:
            if key not in bf_keys:
                raise ValueError(f"{key} missing. {bf_example_config}")

    if "parcellation" in config_keys:
        parc_example_config = """Correct example:
        config = '''
        parcellation:
            file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
        '''"""
        parc_keys = [str(c) for c in config["parcellation"].keys()]
        correct_keys = ["file", "method", "orthogonalisation"]
        for key in parc_keys:
            if key not in correct_keys:
                raise ValueError(f"{key} invalid. {parc_example_config}")
        for key in correct_keys:
            if key not in parc_keys:
                raise ValueError(f"{key} missing. {parc_example_config}")


def run_src_chain(
    config,
    subject,
    preproc_file,
    src_dir,
    smri_file=None,
    pos_file=None,
    edit_polhemus_func=None,
    cleanup_files=True,
    verbose="INFO",
    mneverbose="WARNING",
):
    """Source reconstruction.

    Parameters
    ----------
    config : string or dict
        Source reconstruction config.
    subject : string
        Subject name.
    preproc_file : string
        Preprocessed fif file.
    src_dir : string
        Source reconstruction directory.
    smri_file : string
        Structural MRI file.
    pos_file : string
        Pos files.
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
    verbose : string
        Level of verbose.
    mneverbose : string
        Level of MNE verbose.

    Returns
    -------
    flag : bool
        Flag indicating whether source reconstruction was successful.
    """
    rhino_utils.check_fsl()

    # Directories
    src_dir = validate_outdir(src_dir)
    coreg_dir = validate_outdir(src_dir / "coreg")
    reportdir = validate_outdir(coreg_dir / "report")
    logsdir = validate_outdir(src_dir / "logs")

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

    # Create directory for coregistration and report
    os.makedirs(coreg_dir / subject, exist_ok=True)
    reportdir = validate_outdir(reportdir / run_id)

    # Load config
    if not isinstance(config, dict):
        config = load_config(config)
    _validate_config(config)

    # Validation
    if "coregistration" in config and smri_file is None:
        raise ValueError("smri_file must be passed if we're doing coregistration.")

    # MAIN BLOCK - Run source reconstruction and catch any exceptions
    try:
        # ----------------------------------------------------------------
        # Coregistration
        if "coregistration" in config:
            # Unpack coregistration specific arguments
            include_nose = config["coregistration"]["include_nose"]
            use_nose = config["coregistration"]["use_nose"]
            use_headshape = config["coregistration"]["use_headshape"]
            model = config["coregistration"]["model"]

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
                logger.info(f"edit_polhemus_func: {edit_polhemus_func.__name__}")
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

        # ----------------------------------------------------------------
        # Beamforming
        if "beamforming" in config:
            # Unpack beamforming related arguments
            freq_range = config["beamforming"]["freq_range"]
            chantypes = config["beamforming"]["chantypes"]
            if isinstance(chantypes, str):
                chantypes = [chantypes]
            ranks = config["beamforming"]["ranks"]
            if isinstance(ranks, int):
                ranks = [ranks]
            rank = {}
            for c, r in zip(chantypes, ranks):
                rank.update({c: r})

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
            logger.info(f"channel types and rank for source reconstruction: {rank}")
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

        # ----------------------------------------------------------------
        # Parcellation
        if "parcellation" in config:
            # Unpack parcellation related arguments
            parcellation_file = config["parcellation"]["file"]
            method = config["parcellation"]["method"]
            orthogonalisation = config["parcellation"]["orthogonalisation"]

            current_status = "parcellation"
            logger.info(current_status)
            logger.info(parcellation_file)
            p = parcellation.Parcellation(parcellation_file)

            gridstep = int(rhino_utils.get_gridstep(src_coords_mni.T) / 1000)
            logger.info("gridstep = {} mm".format(gridstep))

            p.parcellate(
                voxel_timeseries=src_ts_mni,
                voxel_coords=src_coords_mni,
                method=method,
                batch_mode=True,
            )
            parcel_ts = p.parcel_timeseries["data"]

            # Orthogonalisation
            if orthogonalisation not in ["symmetric"]:
                raise NotImplementedError(orthogonalisation)

            current_status = f"orthogonalisation: {orthogonalisation}"
            logger.info(current_status)

            if orthogonalisation == "symmetric":
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

    if "coregistration" in config:
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


def run_src_batch(
    config,
    subjects,
    preproc_files,
    src_dir,
    smri_files=None,
    pos_files=None,
    edit_polhemus_func=None,
    cleanup_files=True,
    verbose="INFO",
    mneverbose="WARNING",
    dask_client=False,
):
    """Batch source reconstruction.

    Parameters
    ----------
    config : string or dict
        Source reconstruction config.
    subjects : list of strings
        Subject names.
    preproc_files : list of strings
        Preprocessed fif files.
    src_dir : string
        Source reconstruction directory.
    smri_files : list of strings
        Structural MRI files.
    pos_files : list of strings
        Pos files.
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
    rhino_utils.check_fsl()

    # Directories
    src_dir = validate_outdir(src_dir)
    coreg_dir = validate_outdir(src_dir / "coreg")
    logsdir = validate_outdir(src_dir / "logs")
    reportdir = validate_outdir(coreg_dir / "report")

    # Initialise Loggers
    mne.set_log_level(mneverbose)
    logfile = os.path.join(logsdir, 'osl_batch.log')
    osl_logger.set_up(log_file=logfile, level=verbose, startup=False)
    logger.info('Starting OSL Batch Source Reconstruction')

    # Load config
    config = load_config(config)
    _validate_config(config)
    config_str = pprint.PrettyPrinter().pformat(config)
    logger.info('Running config\n {0}'.format(config_str))

    # Validation
    n_subjects = len(subjects)
    n_preproc_files = len(preproc_files)
    if n_subjects != n_preproc_files:
        raise ValueError(
            "Got {n_subjects} subjects and {n_preproc_files} preproc_files."
        )

    if "coregistration" in config and smri_files is None:
        raise ValueError("smri_files must be passed if we are coregistering.")
    elif smri_files is None:
        smri_files = [None] * n_subjects

    if pos_files is None:
        pos_files = [None] * n_subjects

    # Create partial function with fixed options
    pool_func = partial(
        run_src_chain,
        edit_polhemus_func=edit_polhemus_func,
        cleanup_files=cleanup_files,
        verbose=verbose,
        mneverbose=mneverbose,
    )

    # Loop through input files to generate arguments for run_coreg_chain
    args = []
    for subject, preproc_file, smri_file, pos_file in zip(
        subjects, preproc_files, smri_files, pos_files
    ):
        args.append((
            config, subject, preproc_file, src_dir, smri_file, pos_file
        ))

    # Actually run the processes
    if dask_client:
        flags = parallel.dask_parallel_bag(pool_func, args)
    else:
        flags = [pool_func(*aa) for aa in args]

    logger.info("Processed {0}/{1} files successfully".format(np.sum(flags), len(flags)))

    if "coregistration" in config:
        # Generate HTML report
        raw_report.gen_html_page(reportdir)
        logger.info("******************************" + "*" * len(str(reportdir)))
        logger.info(f"* REMEMBER TO CHECK REPORT: {reportdir} *")
        logger.info("******************************" + "*" * len(str(reportdir)))

    return flags
