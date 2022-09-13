"""Wrappers for source reconstruction.

This module contains the functions callable using a 'source_recon'
section of a config.

All functions in this module accept the following arguments for
consistency:

    func(src_dir, subject, preproc_file, smri_file, *userargs, logger)

Custom functions (i.e. functions passed via the extra_funcs argument)
must also conform to this.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>


import logging
import os.path as op

import numpy as np
from mne.beamformer import apply_lcmv_raw

from . import rhino, beamforming, parcellation


logger = logging.getLogger(__name__)


#------------------------------------------------------------------
# RHINO wrappers


def extract_fiducials_from_fif(
    src_dir, subject, preproc_file, smri_file, logger, **userargs,
):
    # Get coreg filenames
    subjects_dir = op.join(src_dir, "rhino")
    filenames = rhino.get_coreg_filenames(subjects_dir, subject)

    logger.info("Setting up polhemus files")
    rhino.extract_polhemus_from_info(
        fif_file=preproc_file,
        headshape_outfile=filenames["polhemus_headshape_file"],
        nasion_outfile=filenames["polhemus_nasion_file"],
        rpa_outfile=filenames["polhemus_rpa_file"],
        lpa_outfile=filenames["polhemus_lpa_file"],
        **userargs,
    )
    logger.info(f"saved: {filenames['polhemus_headshape_file']}")
    logger.info(f"saved: {filenames['polhemus_nasion_file']}")
    logger.info(f"saved: {filenames['polhemus_rpa_file']}")
    logger.info(f"saved: {filenames['polhemus_lpa_file']}")


def coregister(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    logger,
    include_nose,
    use_nose,
    use_headshape,
    model,
    eeg=False,
):
    subjects_dir = op.join(src_dir, "rhino")

    # Compute surface
    rhino.compute_surfaces(
        smri_file=smri_file,
        subjects_dir=subjects_dir,
        subject=subject,
        include_nose=include_nose,
        logger=logger,
    )

    # Run coregistration
    rhino.coreg(
        fif_file=preproc_file,
        subjects_dir=subjects_dir,
        subject=subject,
        use_headshape=use_headshape,
        use_nose=use_nose,
        logger=logger,
    )

    # Compute forward model
    rhino.forward_model(
        subjects_dir=subjects_dir,
        subject=subject,
        model=model,
        eeg=eeg,
        logger=logger,
    )


#------------------------------------------------------------------
# Beamforming and parcellation wrappers


def beamform_and_parcellate(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    logger,
    chantypes,
    rank,
    freq_range,
    parcellation_file,
    method,
    orthogonalisation,
):
    from ..preprocessing import import_data

    subjects_dir = op.join(src_dir, "rhino")

    # Load preprocessed data
    preproc_data = import_data(preproc_file)
    preproc_data.pick(chantypes)

    if freq_range is not None:
        # Bandpass filter
        logger.info(
            "bandpass filtering: {}-{} Hz".format(
                freq_range[0], freq_range[1]
            )
        )
        preproc_data = preproc_data.filter(
            l_freq=freq_range[0],
            h_freq=freq_range[1],
            method="iir",
            iir_params={"order": 5, "ftype": "butter"},
        )

    # Validation
    if isinstance(chantypes, str):
        chantypes = [chantypes]

    # Create beamforming filters
    logger.info("beamforming.make_lcmv")
    logger.info(f"chantypes: {chantypes}")
    logger.info(f"rank: {rank}")
    filters = beamforming.make_lcmv(
        subjects_dir=subjects_dir,
        subject=subject,
        data=preproc_data,
        chantypes=chantypes,
        weight_norm="nai",
        rank=rank,
        logger=logger,
        save_figs=True,
    )

    # Apply beamforming
    logger.info("mne.beamforming.apply_lcmv")
    src_data = apply_lcmv_raw(preproc_data, filters)
    src_ts_mni, _, src_coords_mni, _ = beamforming.transform_recon_timeseries(
        subjects_dir=subjects_dir,
        subject=subject,
        recon_timeseries=src_data.data,
    )

    # Parcellation
    logger.info("parcellation")
    logger.info(parcellation_file)
    p = parcellation.Parcellation(parcellation_file)
    p.parcellate(
        voxel_timeseries=src_ts_mni,
        voxel_coords=src_coords_mni,
        method=method,
        logger=logger,
    )
    parcel_ts = p.parcel_timeseries["data"]

    # Orthogonalisation
    if orthogonalisation not in [None, "symmetric"]:
        raise NotImplementedError(orthogonalisation)

    if orthogonalisation == "symmetric":
        logger.info(f"{orthogonalisation} orthogonalisation")
        parcel_ts = parcellation.symmetric_orthogonalise(
            parcel_ts, maintain_magnitudes=True
        )

    # Save parcellated data
    parc_data_file = src_dir / f"{subject}.npy"
    logger.info(f"saving {parc_data_file}")
    np.save(parc_data_file, parcel_ts.T)
