#!/usr/bin/env python

"""Beamforming.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import os.path as op

import numpy as np
from mne import (
    read_forward_solution,
    Covariance,
    compute_covariance,
    compute_raw_covariance,
)

from osl.source_recon import rhino, rhino_utils


def make_lcmv(
    subjects_dir,
    subject,
    dat,
    chantypes,
    data_cov=None,
    noise_cov=None,
    reg=0,
    label=None,
    pick_ori="max-power-pre-weight-norm",
    rank="info",
    noise_rank="info",
    weight_norm="unit-noise-gain-invariant",
    reduce_rank=True,
    depth=None,
    inversion="matrix",
    verbose=None,
    batch_mode=False,
):
    """Compute LCMV spatial filter.

    Wrapper for Rhino version of mne.beamformer.make_lcmv

    Parameters
    ----------
    subjects_dir : string
            Directory to find RHINO subject dirs in.
    subject : string
            Subject name dir to find RHINO fwd model file in.
    dat : instance of raw or epochs
        The measurement data to specify the channels to include.
        Bad channels in info['bads'] are not used.
        Will also be used to calculate data_cov
    data_cov : instance of Covariance | None
        The noise covariance matrix used to whiten.
        If None will be computed from dat.
    noise_cov : instance of Covariance | None
        The noise covariance matrix used to whiten.
        If None will be computed from dat as a diagonal matrix
        with variances set to the average of all sensors of that type.
    chantypes : List
        List of channel types to use. E.g.:
            ['eeg']
            ['mag', 'grad']
            ['eeg', 'mag', 'grad']
    reg : float
        The regularization for the whitened data covariance.
    label : instance of Label
        Restricts the LCMV solution to a given label.
    batch_mode : bool
        Are we running in batch mode?

    Returns
    -------
    filters : instance of Beamformer
        Dictionary containing filter weights from LCMV beamformer.
        Contains the following keys:
            'kind' : str
                The type of beamformer, in this case 'LCMV'.
            'weights' : array
                The filter weights of the beamformer.
            'data_cov' : instance of Covariance
                The data covariance matrix used to compute the beamformer.
                If None will be computed from raw.
            'noise_cov' : instance of Covariance | None
                The noise covariance matrix used to whiten.
            'whitener' : None | ndarray, shape (n_channels, n_channels)
                Whitening matrix, provided if whitening was applied to the
                covariance matrix and leadfield during computation of the
                beamformer weights.
            'weight_norm' : str | None
                Type of weight normalization used to compute the filter
                weights.
            'pick-ori' : None | 'max-power' | 'normal' | 'vector'
                The orientation in which the beamformer filters were computed.
            'ch_names' : list of str
                Channels used to compute the beamformer.
            'proj' : array
                Projections used to compute the beamformer.
            'is_ssp' : bool
                If True, projections were applied prior to filter computation.
            'vertices' : list
                Vertices for which the filter weights were computed.
            'is_free_ori' : bool
                If True, the filter was computed with free source orientation.
            'n_sources' : int
                Number of source location for which the filter weight were
                computed.
            'src_type' : str
                Type of source space.
            'source_nn' : ndarray, shape (n_sources, 3)
                For each source location, the surface normal.
            'proj' : ndarray, shape (n_channels, n_channels)
                Projections used to compute the beamformer.
            'subject' : str
                The subject ID.
            'rank' : int
                The rank of the data covariance matrix used to compute the
                beamformer weights.
            'max-power-ori' : ndarray, shape (n_sources, 3) | None
                When pick_ori='max-power', this fields contains the estimated
                direction of maximum ?weight normalised power? at each source location.
                When pick_ori='max-power-pre-weight-norm', this fields contains the estimated
                direction of maximum power at each source location.
            'inversion' : 'single' | 'matrix'
                Whether the spatial filters were computed for each dipole
                separately or jointly for all dipoles at each vertex using a
                matrix inversion.
    """

    if not batch_mode:
        print("\n*** RUNNING OSL MAKE LCMV ***")

    # load forward solution
    fwd_fname = rhino.get_coreg_filenames(subjects_dir, subject)["forward_model_file"]
    fwd = read_forward_solution(fwd_fname)

    is_epoched = len(dat.get_data().shape) == 3 and len(dat) > 1

    if data_cov is None:

        # Note that if chantypes are meg, eeg; and meg includes mag, grad
        # then compute_covariance will project data separately for meg and eeg
        # to reduced rank subspace (i.e. mag and grad will be combined together
        # inside the meg subspace, eeg will haeve a separate subspace).
        # I.e. data will become (ntpts x (rank_meg + rank_eeg))
        # and cov will be (rank_meg + rank_eeg) x (rank_meg + rank_eeg)
        # and include correlations between eeg and meg subspaces.
        # The output data_cov is cov projected back onto the indivdual
        # sensor types mag, grad, eeg.
        #
        # Prior to computing anything, including the subspaces each of mag, grad, eeg
        # are scaled so that they are on comparable scales to aid mixing in the
        # subspace and improve numerical stability. This is equivalent to what the
        # osl_normalise_sensor_data.m function in Matlab OSL is trying to do.
        # Note that in the output data_cov the scalings have been undone.
        if is_epoched:
            data_cov = compute_covariance(dat, method="empirical", rank=rank)
        else:
            data_cov = compute_raw_covariance(dat, method="empirical", rank=rank)

    if noise_cov is None:

        # calculate noise covariance matrix
        # Later this will be inverted and used to whiten the data AND the lead fields
        # as part of the source recon. See:
        #   https://www.sciencedirect.com/science/article/pii/S1053811914010325?via%3Dihub
        #
        # In MNE, the noise cov is normally obtained from empty room noise recordings
        # or from a baseline period.
        # Here (if no noise cov is passed in) we mimic what the
        # osl_normalise_sensor_data.m function in Matlab OSL does,
        # by computing a diagonal noise cov with the variances set to the mean
        # variance of each sensor type (e.g. mag, grad, eeg.)
        variances = {}
        for type in chantypes:
            dat_type = dat.copy().pick(type, exclude="bads")
            noise_cov_diag = np.zeros([data_cov.data.shape[0]])

            inds = []
            ch_names = []
            for ch_name in dat_type.info["ch_names"]:
                inds.append(data_cov.ch_names.index(ch_name))

            variances[type] = np.mean(np.diag(data_cov.data)[inds])
            noise_cov_diag[inds] = variances[type]

            if not batch_mode:
                print("Variance for chan type {} is {}".format(type, variances[type]))

        bads = [b for b in dat.info["bads"] if b in data_cov.ch_names]
        noise_cov = Covariance(
            noise_cov_diag, data_cov.ch_names, bads, dat.info["projs"], nfree=1e10
        )

    filters = rhino_utils._make_lcmv(
        dat.info,
        fwd,
        data_cov,
        noise_cov=noise_cov,
        reg=reg,
        pick_ori=pick_ori,
        weight_norm=weight_norm,
        rank=rank,
        noise_rank=noise_rank,
        reduce_rank=reduce_rank,
        verbose=verbose,
    )

    if not batch_mode:
        print("*** OSL MAKE LCMV COMPLETE ***\n")

    if batch_mode:
        return filters, variances
    else:
        return filters


def get_recon_timeseries(subjects_dir, subject, coord_mni, recon_timeseries_head):
    """Gets the reconstructed time series nearest to the passed in coordinate
    in MNI space>

    Parameters
    ----------
    subjects_dir : string
        Directory to find RHINO subject dirs in.
    subject : string
        Subject name dir to find RHINO files in.
    coord_mni : (3,) np.array
        3D coordinate in MNI space to get timeseries for
    recon_timeseries_head : (ndipoles, ntpts) np.array
        Reconstructed time courses in head (polhemus) space
        Assumes that the dipoles are the same (and in the same order)
        as those in the forward model, coreg_filenames['forward_model_file'].

    Returns
    -------
    recon_timeseries : numpy.ndarray
        The timecourse in recon_timeseries_head nearest to coord_mni
    """

    surfaces_filenames = rhino.get_surfaces_filenames(subjects_dir, subject)
    coreg_filenames = rhino.get_coreg_filenames(subjects_dir, subject)

    # get coord_mni in mri space
    mni_mri_t = rhino_utils.read_trans(surfaces_filenames["mni_mri_t_file"])
    coord_mri = rhino_utils.xform_points(mni_mri_t["trans"], coord_mni)

    # Get hold of coords of points reconstructed to.
    # Note, MNE forward model is done in head space in metres.
    # Rhino does everything in mm
    fwd = read_forward_solution(coreg_filenames["forward_model_file"])
    vs = fwd["src"][0]
    recon_coords_head = vs["rr"][vs["vertno"]] * 1000  # in mm

    # convert coords_head from head to mri space to get index of reconstructed
    # coordinate nearest to coord_mni
    head_mri_t = rhino_utils.read_trans(coreg_filenames["head_mri_t_file"])
    recon_coords_mri = rhino_utils.xform_points(
        head_mri_t["trans"], recon_coords_head.T
    ).T

    recon_index, d = rhino_utils._closest_node(coord_mri.T, recon_coords_mri)

    recon_timeseries = np.abs(recon_timeseries_head[recon_index, :]).T

    return recon_timeseries


def transform_recon_timeseries(
    subjects_dir,
    subject,
    recon_timeseries,
    spatial_resolution=None,
    reference_brain="mni",
    batch_mode=False,
):
    """Spatially resamples a (ndipoles x ntpts) array of reconstructed time
    courses (in head/polhemus space) to dipoles on the brain grid of the specified
    reference brain.

    Parameters
    ----------
    subjects_dir : string
        Directory to find RHINO subject dirs in.
    subject : string
        Subject name dir to find RHINO files in.
    recon_timeseries : numpy.ndarray
        (ndipoles, ntpts) or (ndipoles, ntpts, ntrials) of reconstructed time courses
        (in head (polhemus) space). Assumes that the dipoles are the same (and in the
        same order) as those in the forward model,
        coreg_filenames['forward_model_file']. Typically derive from the
        VolSourceEstimate's output by MNE source recon methods, e.g.
        mne.beamformer.apply_lcmv, obtained using a forward model generated by Rhino.
    spatial_resolution : int
        Resolution to use for the reference brain in mm
        (must be an integer, or will be cast to nearest int)
        If None, then the gridstep used in coreg_filenames['forward_model_file']
        is used.
    reference_brain : string
        'mni' indicates that the reference_brain is the stdbrain in MNI space
        'mri' indicates that the reference_brain is the subject's sMRI in
        native/mri space.
    batch_mode : bool
        Are we in batch mode?

    Returns
    -------
    recon_timeseries_out : numpy.ndarray
        (ndipoles, ntpts) np.array of reconstructed time courses resampled
        on the reference brain grid.
    reference_brain_fname : string
        File name of the requested reference brain at the requested
        spatial resolution, int(spatial_resolution)
        (with zero for background, and !=0 for brain)
    coords_out : numpy.ndarray
        (3, ndipoles) np.array of coordinates (in mm) of dipoles in
        recon_timeseries_out in "reference_brain" space
    """

    surfaces_filenames = rhino.get_surfaces_filenames(subjects_dir, subject)
    coreg_filenames = rhino.get_coreg_filenames(subjects_dir, subject)

    # -------------------------------------------------------
    # Get hold of coords of points reconstructed to.
    # Note, MNE forward model is done in head space in metres.
    # Rhino does everything in mm
    fwd = read_forward_solution(coreg_filenames["forward_model_file"])
    vs = fwd["src"][0]
    recon_coords_head = vs["rr"][vs["vertno"]] * 1000  # in mm

    # -------------------------------------------------------
    if spatial_resolution is None:
        # estimate gridstep from forward model
        rr = fwd["src"][0]["rr"]

        store = []
        for ii in range(rr.shape[0]):
            store.append(np.sqrt(np.sum(np.square(rr[ii, :] - rr[0, :]))))
        store = np.asarray(store)
        spatial_resolution = int(np.round(np.min(store[np.where(store > 0)]) * 1000))
        if not batch_mode:
            print("Using spatial_resolution = {}mm".format(spatial_resolution))

    spatial_resolution = int(spatial_resolution)

    if reference_brain == "mni":
        # reference is mni stdbrain

        # convert recon_coords_head from head to mni space
        head_mri_t = rhino_utils.read_trans(coreg_filenames["head_mri_t_file"])
        recon_coords_mri = rhino_utils.xform_points(
            head_mri_t["trans"], recon_coords_head.T
        ).T

        mni_mri_t = rhino_utils.read_trans(surfaces_filenames["mni_mri_t_file"])
        recon_coords_out = rhino_utils.xform_points(
            np.linalg.inv(mni_mri_t["trans"]), recon_coords_mri.T
        ).T

        reference_brain = (
            os.environ["FSLDIR"] + "/data/standard/MNI152_T1_1mm_brain.nii.gz"
        )

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = op.join(
            coreg_filenames["basefilename"],
            "MNI152_T1_{}mm_brain.nii.gz".format(spatial_resolution),
        )

    elif reference_brain == "mri":
        # reference is smri

        # convert recon_coords_head from head to mri space
        head_mri_t = rhino_utils.read_trans(coreg_filenames["head_mri_t_file"])
        recon_coords_out = rhino_utils.xform_points(
            head_mri_t["trans"], recon_coords_head.T
        ).T

        reference_brain = surfaces_filenames["smri_file"]

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = reference_brain.replace(
            ".nii.gz", "_{}mm.nii.gz".format(spatial_resolution)
        )

    else:
        ValueError("Invalid out_space, should be mni or mri")

    # -------------------------------------------------------------------------
    # get coordinates from reference brain at resolution spatial_resolution

    # create std brain of the required resolution
    rhino_utils.system_call(
        "flirt -in {} -ref {} -out {} -applyisoxfm {}".format(
            reference_brain,
            reference_brain,
            reference_brain_resampled,
            spatial_resolution,
        )
    )

    coords_out, vals = rhino_utils.niimask2mmpointcloud(reference_brain_resampled)

    # -------------------------------------------------------------------------
    # for each mni_coords_out find nearest coord in recon_coords_out

    recon_timeseries_out = np.zeros(
        np.insert(recon_timeseries.shape[1:], 0, coords_out.shape[1])
    )
    #import pdb; pdb.set_trace()

    recon_indices = np.zeros([coords_out.shape[1]])

    for cc in range(coords_out.shape[1]):
        recon_index, dist = rhino_utils._closest_node(
            coords_out[:, cc], recon_coords_out
        )

        if dist < spatial_resolution:
            recon_timeseries_out[cc, :] = recon_timeseries[recon_index, ...]
            recon_indices[cc] = recon_index

    return recon_timeseries_out, reference_brain_resampled, coords_out, recon_indices
