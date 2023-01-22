"""Wrappers for source reconstruction.

This module contains the functions callable using a 'source_recon'
section of a config.

All functions in this module accept the following arguments for
consistency:

    func(src_dir, subject, preproc_file, smri_file, epoch_file, *userargs)

Custom functions (i.e. functions passed via the extra_funcs argument)
must also conform to this.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>


import os.path as op
from pathlib import Path
import pickle

import numpy as np
import mne

from . import rhino, beamforming, parcellation, sign_flipping
from ..report import src_report

import logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# RHINO wrappers


def extract_fiducials_from_fif(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    epoch_file,
    **userargs,
):
    """Wrapper function to extract fiducials/headshape points.

    Parameters
    ----------
    src_dir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    smri_file : str
        Path to the T1 weighted structural MRI file to use in source reconstruction.
    epoch_file : str
        Path to epoched preprocessed fif file.
    userargs : keyword arguments
        Keyword arguments to pass to
        osl.source_recon.rhino.extract_polhemus_from_info.
    """
    filenames = rhino.get_coreg_filenames(src_dir, subject)

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


def compute_surfaces(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    epoch_file,
    include_nose,
    recompute_surfaces=False,
):
    """Wrapper for computing surfaces.

    Parameters
    ----------
    src_dir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    smri_file : str
        Path to the T1 weighted structural MRI file to use in source reconstruction.
    epoch_file : str
        Path to epoched preprocessed fif file.
    include_nose : bool
        Should we include the nose when we're extracting the surfaces?
    recompute_surfaces : bool
        Specifies whether or not to run compute_surfaces, if the passed in
        options have already been run
    """
    # Compute surfaces
    rhino.compute_surfaces(
        smri_file=smri_file,
        subjects_dir=src_dir,
        subject=subject,
        include_nose=include_nose,
        recompute_surfaces=recompute_surfaces,
    )

    # Save info for the report
    src_report.add_to_data(
        f"{src_dir}/{subject}/report_data.pkl",
        {
            "compute_surfaces": True,
            "include_nose": include_nose,
        }
    )


def coreg(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    epoch_file,
    use_nose,
    use_headshape,
    already_coregistered=False,
    allow_smri_scaling=False,
    n_init=30,
):
    """Wrapper for full coregistration: compute_surfaces, coreg and forward_model.

    Parameters
    ----------
    src_dir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    smri_file : str
        Path to the T1 weighted structural MRI file to use in source reconstruction.
    epoch_file : str
        Path to epoched preprocessed fif file.
    use_nose : bool
        Should we use the nose in the coregistration?
    use_headshape : bool
        Should we use the headshape points in the coregistration?
    already_coregistered : bool
        Indicates that the data is already coregistered.
    allow_smri_scaling : bool
        Indicates if we are to allow scaling of the sMRI, such that the sMRI-derived fids
        are scaled in size to better match the polhemus-derived fids.
        This assumes that we trust the size (e.g. in mm) of the polhemus-derived fids,
        but not the size of the sMRI-derived fids.
        E.g. this might be the case if we do not trust the size (e.g. in mm) of the sMRI,
        or if we are using a template sMRI that has not come from this subject.
    n_init : int
        Number of initialisation for coregistration.
    """
    # Run coregistration
    rhino.coreg(
        fif_file=preproc_file,
        subjects_dir=src_dir,
        subject=subject,
        use_headshape=use_headshape,
        use_nose=use_nose,
        already_coregistered=already_coregistered,
        allow_smri_scaling=allow_smri_scaling,
        n_init=n_init,
    )

    # Calculate metrics
    if already_coregistered:
        fid_err = None
    else:
        fid_err = rhino.coreg_metrics(subjects_dir=src_dir, subject=subject)

    # Save plots
    rhino.coreg_display(
        subjects_dir=src_dir,
        subject=subject,
        display_outskin_with_nose=False,
        filename=f"{src_dir}/{subject}/rhino/coreg.html",
    )
    
    # Save info for the report
    src_report.add_to_data(
        f"{src_dir}/{subject}/report_data.pkl",
        {
            "coreg": True,
            "use_headshape": use_headshape,
            "use_nose": use_nose,
            "already_coregistered": already_coregistered,
            "allow_smri_scaling": allow_smri_scaling,
            "n_init_coreg": n_init,
            "fid_err": fid_err,
            "coreg_plot": f"{src_dir}/{subject}/rhino/coreg.html",
        }
    )


def forward_model(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    epoch_file,
    model,
    eeg=False,
):
    """Wrapper for computing the forward model.

    Parameters
    ----------
    src_dir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    smri_file : str
        Path to the T1 weighted structural MRI file to use in source reconstruction.
    epoch_file : str
        Path to epoched preprocessed fif file.
    eeg : bool
        Are we using EEG channels in the source reconstruction?
    """
    # Compute forward model
    rhino.forward_model(
        subjects_dir=src_dir,
        subject=subject,
        model=model,
        eeg=eeg,
    )

    # Save info for the report
    src_report.add_to_data(
        f"{src_dir}/{subject}/report_data.pkl",
        {
            "forward_model": True,
            "model": model,
            "eeg": eeg,
        }
    )


def coregister(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    epoch_file,
    include_nose,
    use_nose,
    use_headshape,
    model,
    recompute_surfaces=False,
    already_coregistered=False,
    allow_smri_scaling=False,
    n_init=30,
    eeg=False,
):
    """Wrapper for full coregistration: compute_surfaces, coreg and forward_model.

    Parameters
    ----------
    src_dir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    smri_file : str
        Path to the T1 weighted structural MRI file to use in source reconstruction.
    epoch_file : str
        Path to epoched preprocessed fif file.
    include_nose : bool
        Should we include the nose when we're extracting the surfaces?
    use_nose : bool
        Should we use the nose in the coregistration?
    use_headshape : bool
        Should we use the headshape points in the coregistration?
    model : str
        Forward model to use.
    recompute_surfaces : bool
        Specifies whether or not to run compute_surfaces, if the passed in
        options have already been run
    already_coregistered : bool
        Indicates that the data is already coregistered.
    allow_smri_scaling : bool
        Indicates if we are to allow scaling of the sMRI, such that the sMRI-derived fids
        are scaled in size to better match the polhemus-derived fids.
        This assumes that we trust the size (e.g. in mm) of the polhemus-derived fids,
        but not the size of the sMRI-derived fids.
        E.g. this might be the case if we do not trust the size (e.g. in mm) of the sMRI,
        or if we are using a template sMRI that has not come from this subject.
    eeg : bool
        Are we using EEG channels in the source reconstruction?
    n_init : int
        Number of initialisation for coregistration.
    """
    # Compute surfaces
    rhino.compute_surfaces(
        smri_file=smri_file,
        subjects_dir=src_dir,
        subject=subject,
        include_nose=include_nose,
        recompute_surfaces=recompute_surfaces,
    )

    # Run coregistration
    rhino.coreg(
        fif_file=preproc_file,
        subjects_dir=src_dir,
        subject=subject,
        use_headshape=use_headshape,
        use_nose=use_nose,
        already_coregistered=already_coregistered,
        allow_smri_scaling=allow_smri_scaling,
        n_init=n_init,
    )

    # Calculate metrics
    if already_coregistered:
        fid_err = None
    else:
        fid_err = rhino.coreg_metrics(subjects_dir=src_dir, subject=subject)

    # Save plots
    rhino.coreg_display(
        subjects_dir=src_dir,
        subject=subject,
        display_outskin_with_nose=False,
        filename=f"{src_dir}/{subject}/rhino/coreg.html",
    )

    # Compute forward model
    rhino.forward_model(
        subjects_dir=src_dir,
        subject=subject,
        model=model,
        eeg=eeg,
    )

    # Save info for the report
    src_report.add_to_data(
        f"{src_dir}/{subject}/report_data.pkl",
        {
            "coregister": True,
            "compute_surfaces": True,
            "coreg": True,
            "forward_model": True,
            "include_nose": include_nose,
            "use_nose": use_nose,
            "use_headshape": use_headshape,
            "already_coregistered": already_coregistered,
            "allow_smri_scaling": allow_smri_scaling,
            "n_init_coreg": n_init,
            "forward_model": True,
            "model": model,
            "eeg": eeg,
            "fid_err": fid_err,
            "coreg_plot": f"{src_dir}/{subject}/rhino/coreg.html",
        }
    )


# ------------------------------------------------------------------
# Beamforming and parcellation wrappers


def beamform(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    epoch_file,
    freq_range,
    chantypes,
    rank,
    spatial_resolution=None,
    reference_brain="mni",
    save_bf_data=False,
):
    """Wrapper function for beamforming.

    Parameters
    ----------
    src_dir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    smri_file : str
        Path to the T1 weighted structural MRI file to use in source reconstruction.
    epoch_file : str
        Path to epoched preprocessed fif file.
    freq_range : list
        Lower and upper band to bandpass filter before beamforming. If None,
        no filtering is done.
    chantypes : list of str
        Channel types to use in beamforming.
    rank : dict
        Keys should be the channel types and the value should be the rank to use.
    spatial_resolution : int
        Resolution for beamforming to use for the reference brain in mm
        (must be an integer, or will be cast to nearest int)
        If None, then the gridstep used in coreg_filenames['forward_model_file']
        is used.
    reference_brain : string
        'mni' indicates that the reference_brain is the stdbrain in MNI space
        'mri' indicates that the reference_brain is the subject's sMRI in
            the scaled native/mri space. "
        'unscaled_mri' indicates that the reference_brain is the subject's sMRI in
            unscaled native/mri space.
        Note that Scaled/unscaled relates to the allow_smri_scaling option in coreg.
        If allow_scaling was False, then the unscaled MRI will be the same as the scaled.
        MRI.
    save_bf_data : bool
        Should we save the beamformed data?
    """
    logger.info("beamforming")

    if epoch_file is not None:
        logger.info("using epoched data")

        # Load epoched data
        data = mne.read_epochs(epoch_file, preload=True)
    else:
        # Load preprocessed data
        data = mne.io.read_raw_fif(preproc_file, preload=True)
    data.pick(chantypes)

    if freq_range is not None:
        # Bandpass filter
        logger.info("bandpass filtering: {}-{} Hz".format(freq_range[0], freq_range[1]))
        data = data.filter(
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
        subjects_dir=src_dir,
        subject=subject,
        data=data,
        chantypes=chantypes,
        weight_norm="nai",
        rank=rank,
        save_figs=True,
    )

    # Save the beamforming filter
    filters_file = src_dir / subject / "rhino/filters-lcmv.h5"
    filters.save(filters_file, overwrite=True)

    # Apply beamforming
    logger.info("beamforming.apply_lcmv")
    bf_data = beamforming.apply_lcmv(data, filters)
    if epoch_file is not None:
        bf_data = np.transpose([bf.data for bf in bf_data], axes=[1, 2, 0])
    else:
        bf_data = bf_data.data
    bf_data_mni, _, _, _ = beamforming.transform_recon_timeseries(
        subjects_dir=src_dir,
        subject=subject,
        recon_timeseries=bf_data,
        spatial_resolution=spatial_resolution,
        reference_brain=reference_brain,
    )

    if save_bf_data:
        # Save beamformed data
        bf_data_file = src_dir / subject / "rhino/bf.npy"
        logger.info(f"saving {bf_data_file}")
        np.save(bf_data_file, bf_data_mni.T)

    # Save info for the report
    src_report.add_to_data(
        f"{src_dir}/{subject}/report_data.pkl",
        {
            "beamform": True,
            "chantypes": chantypes,
            "rank": rank,
            "freq_range": freq_range,
            "filter_cov_plot": f"{src_dir}/{subject}/rhino/filter_cov.png",
            "filter_svd_plot": f"{src_dir}/{subject}/rhino/filter_svd.png",
        }
    )


def beamform_and_parcellate(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    epoch_file,
    chantypes,
    rank,
    freq_range,
    parcellation_file,
    method,
    orthogonalisation,
    spatial_resolution=None,
    reference_brain="mni",
    save_bf_data=False,
):
    """Wrapper function for beamforming and parcellation.

    Parameters
    ----------
    src_dir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    smri_file : str
        Path to the T1 weighted structural MRI file to use in source reconstruction.
    epoch_file : str
        Path to epoched preprocessed fif file.
    chantypes : list of str
        Channel types to use in beamforming.
    rank : dict
        Keys should be the channel types and the value should be the rank to use.
    freq_range : list
        Lower and upper band to bandpass filter before beamforming. If None,
        no filtering is done.
    parcellation_file : str
        Path to the parcellation file to use.
    method : str
        Method to use in the parcellation.
    orthogonalisation : bool
        Should we do orthogonalisation?
    spatial_resolution : int
        Resolution for beamforming to use for the reference brain in mm
        (must be an integer, or will be cast to nearest int)
        If None, then the gridstep used in coreg_filenames['forward_model_file']
        is used.
    reference_brain : string
        'mni' indicates that the reference_brain is the stdbrain in MNI space
        'mri' indicates that the reference_brain is the subject's sMRI in
            the scaled native/mri space. "
        'unscaled_mri' indicates that the reference_brain is the subject's sMRI in
            unscaled native/mri space.
        Note that Scaled/unscaled relates to the allow_smri_scaling option in coreg.
        If allow_scaling was False, then the unscaled MRI will be the same as the scaled.
        MRI.
    save_bf_data : bool
        Should we save the beamformed data?
    """
    logger.info("beamform_and_parcellate")

    if epoch_file is not None:
        logger.info("using epoched data")

        # Load epoched data
        data = mne.read_epochs(epoch_file, preload=True)
    else:
        # Load preprocessed data
        data = mne.io.read_raw_fif(preproc_file, preload=True)
    data.pick(chantypes)

    if freq_range is not None:
        # Bandpass filter
        logger.info("bandpass filtering: {}-{} Hz".format(freq_range[0], freq_range[1]))
        data = data.filter(
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
        subjects_dir=src_dir,
        subject=subject,
        data=data,
        chantypes=chantypes,
        weight_norm="nai",
        rank=rank,
        save_figs=True,
    )

    # Save the beamforming filter
    filters_file = src_dir / subject / "rhino/filters-lcmv.h5"
    filters.save(filters_file, overwrite=True)

    # Apply beamforming
    logger.info("beamforming.apply_lcmv")
    bf_data = beamforming.apply_lcmv(data, filters)
    if epoch_file is not None:
        bf_data = np.transpose([bf.data for bf in bf_data], axes=[1, 2, 0])
    else:
        bf_data = bf_data.data
    bf_data_mni, _, coords_mni, _ = beamforming.transform_recon_timeseries(
        subjects_dir=src_dir,
        subject=subject,
        recon_timeseries=bf_data,
        spatial_resolution=spatial_resolution,
        reference_brain=reference_brain,
    )

    if save_bf_data:
        # Save beamformed data
        bf_data_file = src_dir / subject / "rhino/bf.npy"
        logger.info(f"saving {bf_data_file}")
        np.save(bf_data_file, bf_data_mni.T)

    # Parcellation
    logger.info("parcellation")
    logger.info(f"using file {parcellation_file}")
    parcel_data, _, _ = parcellation.parcellate_timeseries(
        parcellation_file,
        voxel_timeseries=bf_data_mni,
        voxel_coords=coords_mni,
        method=method,
        working_dir=src_dir / subject / "rhino",
    )

    # Orthogonalisation
    if orthogonalisation not in [None, "symmetric", "none", "None"]:
        raise NotImplementedError(orthogonalisation)

    if orthogonalisation == "symmetric":
        logger.info(f"{orthogonalisation} orthogonalisation")
        parcel_data = parcellation.symmetric_orthogonalise(
            parcel_data, maintain_magnitudes=True
        )

    # Create and save mne raw object for the parcellated data
    parc_fif_file = src_dir / subject / "rhino/parc-raw.fif"
    parc_raw = parcellation.convert2mne_raw(parcel_data.T, data)
    parc_raw.save(parc_fif_file, overwrite=True)

    # Save parcellated data
    parc_data_file = src_dir / subject / "rhino/parc.npy"
    logger.info(f"saving {parc_data_file}")
    np.save(parc_data_file, parcel_data.T)

    # Save plots
    parcellation.plot_correlation(
        parcel_data, filename=f"{src_dir}/{subject}/rhino/parc_corr.png",
    )

    # Save info for the report
    n_parcels = parcel_data.shape[0]
    n_samples = parcel_data.shape[1]
    if parcel_data.ndim == 3:
        n_epochs = parcel_data.shape[2]
    else:
        n_epochs = None
    src_report.add_to_data(
        f"{src_dir}/{subject}/report_data.pkl",
        {
            "beamform_and_parcellate": True,
            "beamform": True,
            "parcellate": True,
            "chantypes": chantypes,
            "rank": rank,
            "freq_range": freq_range,
            "filter_cov_plot": f"{src_dir}/{subject}/rhino/filter_cov.png",
            "filter_svd_plot": f"{src_dir}/{subject}/rhino/filter_svd.png",
            "parcellation_file": parcellation_file,
            "method": method,
            "orthogonalisation": orthogonalisation,
            "parc_data_file": str(parc_data_file),
            "n_samples": n_samples,
            "n_parcels": n_parcels,
            "n_epochs": n_epochs,
            "parc_corr_plot": f"{src_dir}/{subject}/rhino/parc_corr.png",
        }
    )


# ------------------------------------------------------------------
# Sign flipping wrappers


def find_template_subject(src_dir, subjects, n_embeddings=1, standardize=True):
    """Function to find a good subject to align other subjects to in the sign flipping.

    Note, this function expects parcellated data to exist in the following location:
    src_dir/*/rhino/parc.npy, the * here represents subject directories.

    Parameters
    ----------
    src_dir : str
        Path to where to output the source reconstruction files.
    subjects : str
        Subjects to include.
    n_embeddings : int
        Number of time-delay embeddings that we will use (if we are doing any).
    standardize : bool
        Should we standardize (z-transform) the data before sign flipping?

    Returns
    -------
    template : str
        Template subject.
    """
    print("Finding template subject:")

    # Get the parcellated data files
    parc_files = []
    for subject in subjects:
        parc_file = op.join(src_dir, subject, "rhino", "parc.npy")
        if Path(parc_file).exists():
            parc_files.append(parc_file)
        else:
            print(f"Warning: {parc_file} not found")

    # Validation
    n_parc_files = len(parc_files)
    if n_parc_files < 2:
        raise ValueError(
            "two or more parcellated data files are needed to perform "
            + f"sign flipping, got {n_parc_files}"
        )

    # Calculate the covariance matrix of each subject
    covs = sign_flipping.load_covariances(parc_files, n_embeddings, standardize)

    # Find a subject to use as a template
    template_index = sign_flipping.find_template_subject(covs, n_embeddings)
    template_subject = parc_files[template_index].split("/")[-3]
    print("Template for sign flipping:", template_subject)

    return template_subject


def fix_sign_ambiguity(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    epoch_file,
    template,
    n_embeddings,
    standardize,
    n_init,
    n_iter,
    max_flips,
):
    """Wrapper function for fixing the dipole sign ambiguity.

    Parameters
    ----------
    src_dir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    smri_file : str
        Path to the T1 weighted structural MRI file to use in source reconstruction.
    epoch_file : str
        Path to epoched preprocessed fif file.
    template : str
        Template subject.
    n_embeddings : int
        Number of time-delay embeddings that we will use (if we are doing any).
    standardize : bool
        Should we standardize (z-transform) the data before sign flipping?
    n_init : int
        Number of initializations.
    n_iter : int
        Number of sign flipping iterations per subject to perform.
    max_flips : int
        Maximum number of channels to flip in an iteration.
    """
    logger.info("fix_sign_ambiguity")
    logger.info(f"using template: {template}")

    # Get path to the parcellated data file for this subject and the template
    parc_files = []
    for sub in [subject, template]:
        parc_file = op.join(src_dir, str(sub), "rhino", "parc.npy")
        if not Path(parc_file).exists():
            raise ValueError(f"{parc_file} not found")
        parc_files.append(parc_file)

    # Calculate the covariance of this subject and the template
    [cov, template_cov] = sign_flipping.load_covariances(
        parc_files, n_embeddings, standardize, use_tqdm=False
    )

    # Find the channels to flip
    flips, metrics = sign_flipping.find_flips(
        cov,
        template_cov,
        n_embeddings,
        n_init,
        n_iter,
        max_flips,
        use_tqdm=False,
    )

    # Apply flips to the parcellated data
    sign_flipping.apply_flips(src_dir, subject, flips)

    # Save info for the report
    src_report.add_to_data(
        f"{src_dir}/{subject}/report_data.pkl",
        {
            "fix_sign_ambiguity": True,
            "template": template,
            "n_embeddings": n_embeddings,
            "standardize": standardize,
            "n_init": n_init,
            "n_iter": n_iter,
            "max_flips": max_flips,
            "metrics": metrics,
        }
    )

