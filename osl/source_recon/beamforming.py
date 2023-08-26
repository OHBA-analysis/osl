"""Beamforming.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import os.path as op

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import read_forward_solution, Covariance, compute_covariance, compute_raw_covariance
from mne.io.meas_info import _simplify_info
from mne.io.pick import pick_channels_cov, pick_info
from mne.io.proj import make_projector
from mne.rank import compute_rank
from mne.minimum_norm.inverse import _check_depth, _prepare_forward, _get_vertno
from mne.source_estimate import _get_src_type
from mne.forward import _subject_from_forward
from mne.forward.forward import is_fixed_orient
from mne.beamformer import read_beamformer
from mne.beamformer._lcmv import _apply_lcmv
from mne.beamformer._compute_beamformer import _reduce_leadfield_rank, _sym_inv_sm, Beamformer
from mne.minimum_norm.inverse import _check_reference
from mne.utils import (
    _check_channels_spatial_filter,
    _check_one_ch_type,
    _check_info_inv,
    _check_option,
    _reg_pinv,
    _pl,
    _sym_mat_pow,
    _check_src_normal,
    check_version,
    verbose,
    warn,
)
from mne.utils import logger as mne_logger

from osl.source_recon import rhino
from osl.source_recon.rhino import utils as rhino_utils
from osl.utils.logger import log_or_print


def get_beamforming_filenames(subjects_dir, subject):
    """Get beamforming filenames.

    Files will be in subjects_dir/subject/beamforming

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories.
    subject : string
        Subject name.

    Returns
    -------
    filenames : dict
        A dict of files.
    """
    basedir = op.join(subjects_dir, subject, "beamform")
    if " " in basedir:
        raise ValueError("subjects_dir/src_dir cannot contain spaces.")
    os.makedirs(basedir, exist_ok=True)
    
    filenames = {
        "basedir": basedir,
        "filters_file": op.join(basedir, "filters-lcmv.h5"),
        "filters_plot_cov": op.join(basedir, "filters_cov.png"),
        "filters_plot_svd": op.join(basedir, "filters_svd.png"),
    }
    return filenames


def make_lcmv(
    subjects_dir,
    subject,
    data,
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
    save_filters=False,
):
    """Compute LCMV spatial filter.

    Modified version of mne.beamformer.make_lcmv.

    Parameters
    ----------
    subjects_dir : string
        Directory to find subject directories in.
    subject : string
        Subject name.
    data : instance of mne.Raw | mne.Epochs
        The measurement data to specify the channels to include. Bad channels in info['bads'] are not used. Will also be used to calculate data_cov.
    data_cov : instance of mne.Covariance | None
        The noise covariance matrix used to whiten. If None will be computed from dat.
    noise_cov : instance of mne.Covariance | None
        The noise covariance matrix used to whiten. If None will be computed from dat as a diagonal matrix with variances set to the average of all sensors of that type.
    chantypes : list
        List of channel types to use. E.g. ['eeg'], ['mag', 'grad'], ['eeg', 'mag', 'grad'].
    reg : float
        The regularization for the whitened data covariance.
    label : instance of Label
        Restricts the LCMV solution to a given label.
    save_filters : bool
        Should we save the LCMV beamforming filter?

    Returns
    -------
    filters : instance of mne.beamformer.Beamformer
        Dictionary containing filter weights from LCMV beamformer. See MNE docs.
    """
    log_or_print("*** RUNNING OSL MAKE LCMV ***")

    rhino_files = rhino_utils.get_rhino_files(subjects_dir, subject)
    beamforming_files = get_beamforming_filenames(subjects_dir, subject)

    # load forward solution
    fwd_fname = rhino_files["fwd_model"]
    fwd = read_forward_solution(fwd_fname)

    if data_cov is None:
        # Note that if chantypes are meg, eeg; and meg includes mag, grad then compute_covariance will project data separately for meg and eeg to reduced
        # rank subspace (i.e. mag and grad will be combined together inside the meg subspace, eeg will haeve a separate subspace). I.e. data will become
        # (ntpts x (rank_meg + rank_eeg)) and cov will be (rank_meg + rank_eeg) x (rank_meg + rank_eeg) and include correlations between eeg and meg subspaces.
        # The output data_cov is cov projected back onto the indivdual sensor types mag, grad, eeg.
        #
        # Prior to computing anything, including the subspaces each of mag, grad, eeg are scaled so that they are on comparable scales to aid mixing in the
        # subspace and improve numerical stability. This is equivalent to what the osl_normalise_sensor_data.m function in Matlab OSL is trying to do.
        # Note that in the output data_cov the scalings have been undone.
        if isinstance(data, mne.io.Raw):
            data_cov = compute_raw_covariance(data, method="empirical", rank=rank)
        else:
            data_cov = compute_covariance(data, method="empirical", rank=rank)

    if noise_cov is None:
        # Calculate noise covariance matrix
        #
        # Later this will be inverted and used to whiten the data AND the lead fields as part of the source recon. See:
        #
        #   https://www.sciencedirect.com/science/article/pii/S1053811914010325?via%3Dihub
        #
        # In MNE, the noise cov is normally obtained from empty room noise recordings or from a baseline period. Here (if no noise cov is passed in) we
        # mimic what the osl_normalise_sensor_data.m function in Matlab OSL does, by computing a diagonal noise cov with the variances set to the mean
        # variance of each sensor type (e.g. mag, grad, eeg.)
        n_channels = data_cov.data.shape[0]
        noise_cov_diag = np.zeros(n_channels)
        for type in chantypes:
            # Indices of this channel type
            type_data = data.copy().pick(type, exclude="bads")
            inds = []
            for chan in type_data.info["ch_names"]:
                inds.append(data_cov.ch_names.index(chan))

            # Mean variance of channels of this type
            variance = np.mean(np.diag(data_cov.data)[inds])
            noise_cov_diag[inds] = variance
            log_or_print("variance for chantype {} is {}".format(type, variance))

        bads = [b for b in data.info["bads"] if b in data_cov.ch_names]
        noise_cov = Covariance(noise_cov_diag, data_cov.ch_names, bads, data.info["projs"], nfree=1e10)

    filters = _make_lcmv(
        data.info,
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

    if save_filters:
        log_or_print(f"saving {beamforming_files['filters_file']}")
        filters.save(beamforming_files["filters_file"], overwrite=True)

    log_or_print("*** OSL MAKE LCMV COMPLETE ***")

    return filters


def make_plots(subjects_dir, subject, filters, data):
    """Plot LCMV beamforming filters.

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories.
    subject : string
        Subject name.
    filters : mne.beamformer.Beamformer
        Filters to plot.
    data : mne.Raw or mne.Epochs
        Data used to create the filters.

    Returns
    -------
    filters_cov_plot : str
        Path to covariance plot.
    filters_svd_plot : str
        Path to eigenspectrum plot.
    """
    beamforming_files = get_beamforming_filenames(subjects_dir, subject)
    fig_cov, fig_svd = filters["data_cov"].plot(data.info, show=False, verbose=False)
    fig_cov.savefig(beamforming_files["filters_plot_cov"], dpi=150)
    fig_svd.savefig(beamforming_files["filters_plot_svd"], dpi=150)
    plt.close("all")
    return beamforming_files["filters_plot_cov"], beamforming_files["filters_plot_svd"]


def load_lcmv(subjects_dir, subject):
    """Load LCMV beamforming filters.

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories.
    subject : string
        Subject name.

    Returns
    -------
    filters : instance of mne.beamformer.Beamformer
        Dictionary containing filter weights from LCMV beamformer. See MNE docs.
    """
    filenames = get_beamforming_filenames(subjects_dir, subject)
    log_or_print(f"loading {filenames['filters_file']}")
    return read_beamformer(filenames["filters_file"])


def apply_lcmv(data, filters, reject_by_annotation="omit"):
    """Apply a LCMV filter to an MNE Raw or Epochs object."""
    log_or_print("beamforming.apply_lcmv")
    if isinstance(data, mne.io.Raw):
        return apply_lcmv_raw(data, filters, reject_by_annotation)
    else:
        return mne.beamformer.apply_lcmv_epochs(data, filters)


def apply_lcmv_raw(raw, filters, reject_by_annotation="omit"):
    """Modified version of mne.beamformer.apply_lcmv_raw."""

    _check_reference(raw)

    # Get data from the mne.Raw object
    data, times = raw.get_data(return_times=True, reject_by_annotation=reject_by_annotation)

    # Select channels
    chan_inds = _check_channels_spatial_filter(raw.ch_names, filters)
    data = data[chan_inds]

    # Apply LCMV beamformer
    stc = _apply_lcmv(data=data, filters=filters, info=raw.info, tmin=times[0])

    return next(stc)


def get_recon_timeseries(subjects_dir, subject, coord_mni, recon_timeseries_head):
    """Gets the reconstructed time series nearest to the passed coordinates in MNI space.

    Parameters
    ----------
    subjects_dir : string
        Directory to find subject directories in.
    subject : string
        Subject name.
    coord_mni : (3,) numpy.ndarray
        3D coordinate in MNI space to get timeseries for
    recon_timeseries_head : (ndipoles, ntpts) np.array
        Reconstructed time courses in head (polhemus) space. Assumes that the dipoles are the same (and in the same order) as those in the forward model, rhino_files['fwd_model'].

    Returns
    -------
    recon_timeseries : numpy.ndarray
        The timecourse in recon_timeseries_head nearest to coord_mni.
    """

    rhino_files = rhino_utils.get_rhino_files(subjects_dir, subjects)
    surfaces_filenames = rhino_files["surf"]
    coreg_filenames = rhino_files["coreg"]

    # Get coord_mni in mri space
    mni_mri_t = rhino_utils.read_trans(surfaces_filenames["mni_mri_t_file"])
    coord_mri = rhino_utils.xform_points(mni_mri_t["trans"], coord_mni)

    # Get hold of coords of points reconstructed to. Note, MNE forward model is done in head space in metres. RHINO does everything in mm
    fwd = read_forward_solution(rhino_files["fwd_model"])
    vs = fwd["src"][0]
    recon_coords_head = vs["rr"][vs["vertno"]] * 1000  # in mm

    # Convert coords_head from head to mri space to get index of reconstructed coordinate nearest to coord_mni
    head_scaledmri_t = rhino_utils.read_trans(coreg_filenames["head_scaledmri_t_file"])
    recon_coords_scaledmri = rhino_utils.xform_points(head_scaledmri_t["trans"], recon_coords_head.T).T

    recon_index, d = rhino_utils._closest_node(coord_mri.T, recon_coords_scaledmri)

    recon_timeseries = np.abs(recon_timeseries_head[recon_index, :]).T

    return recon_timeseries


def transform_recon_timeseries(
    subjects_dir,
    subject,
    recon_timeseries,
    spatial_resolution=None,
    reference_brain="mni",
):
    """Spatially resamples a (ndipoles x ntpts) array of reconstructed time courses (in head/polhemus space) to dipoles on the brain grid of the specified reference brain.

    Parameters
    ----------
    subjects_dir : string
        Directory to find subject directories in.
    subject : string
        Subject name.
    recon_timeseries : numpy.ndarray
        (ndipoles, ntpts) or (ndipoles, ntpts, ntrials) of reconstructed time courses (in head (polhemus) space). Assumes that the dipoles are the same (and in the
        same order) as those in the forward model, rhino_files['fwd_model']. Typically derive from the VolSourceEstimate's output by MNE source recon methods,
        e.g. mne.beamformer.apply_lcmv, obtained using a forward model generated by RHINO.
    spatial_resolution : int
        Resolution to use for the reference brain in mm (must be an integer, or will be cast to nearest int). If None, then the gridstep used in rhino_files['fwd_model'] is used.
    reference_brain : string
        'mni' indicates that the reference_brain is the stdbrain in MNI space.
        'mri' indicates that the reference_brain is the subject's sMRI in the scaled native/mri space.
        'unscaled_mri' indicates that the reference_brain is the subject's sMRI in unscaled native/mri space.
        Note that scaled/unscaled relates to the allow_smri_scaling option in coreg. If allow_scaling was False, then the unscaled MRI will be the same as the scaled MRI.

    Returns
    -------
    recon_timeseries_out : (ndipoles, ntpts) numpy.ndarray
        Array of reconstructed time courses resampled on the reference brain grid.
    reference_brain_fname : string
        File name of the requested reference brain at the requested spatial resolution, int(spatial_resolution) (with zero for background, and !=0 for brain).
    coords_out : (3, ndipoles) numpy.ndarray
        Array of coordinates (in mm) of dipoles in recon_timeseries_out in "reference_brain" space.
    """

    rhino_files = rhino_utils.get_rhino_files(subjects_dir, subject)
    surfaces_filenames = rhino_files["surf"]
    coreg_filenames = rhino_files["coreg"]

    # -------------------------------------------------------------------------------------
    # Get hold of coords of points reconstructed to
    #
    # Note, MNE forward model is done in head space in metres. RHINO does everything in mm.
    fwd = read_forward_solution(rhino_files["fwd_model"])
    vs = fwd["src"][0]
    recon_coords_head = vs["rr"][vs["vertno"]] * 1000  # in mm

    # ----------------------------
    if spatial_resolution is None:
        # Estimate gridstep from forward model
        rr = fwd["src"][0]["rr"]

        store = []
        for ii in range(rr.shape[0]):
            store.append(np.sqrt(np.sum(np.square(rr[ii, :] - rr[0, :]))))
        store = np.asarray(store)
        spatial_resolution = int(np.round(np.min(store[np.where(store > 0)]) * 1000))

    spatial_resolution = int(spatial_resolution)
    log_or_print(f"spatial_resolution = {spatial_resolution} mm")

    if reference_brain == "mni":
        # Reference is mni stdbrain

        # Convert recon_coords_head from head to mni space, head_mri_t_file xform is to unscaled MRI
        head_mri_t = rhino_utils.read_trans(coreg_filenames["head_mri_t_file"])
        recon_coords_mri = rhino_utils.xform_points(head_mri_t["trans"], recon_coords_head.T).T

        # mni_mri_t_file xform is to unscaled MRI
        mni_mri_t = rhino_utils.read_trans(surfaces_filenames["mni_mri_t_file"])
        recon_coords_out = rhino_utils.xform_points(np.linalg.inv(mni_mri_t["trans"]), recon_coords_mri.T).T

        reference_brain = os.environ["FSLDIR"] + "/data/standard/MNI152_T1_1mm_brain.nii.gz"

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = op.join(coreg_filenames["basedir"], "MNI152_T1_{}mm_brain.nii.gz".format(spatial_resolution))

    elif reference_brain == "unscaled_mri":
        # Reference is unscaled smri

        # convert recon_coords_head from head to mri space
        head_mri_t = rhino_utils.read_trans(coreg_filenames["head_mri_t_file"])
        recon_coords_out = rhino_utils.xform_points(head_mri_t["trans"], recon_coords_head.T).T

        reference_brain = surfaces_filenames["smri_file"]

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = reference_brain.replace(".nii.gz", "_{}mm.nii.gz".format(spatial_resolution))

    elif reference_brain == "mri":
        # Reference is scaled smri

        # Convert recon_coords_head from head to mri space
        head_scaledmri_t = rhino_utils.read_trans(coreg_filenames["head_scaledmri_t_file"])
        recon_coords_out = rhino_utils.xform_points(head_scaledmri_t["trans"], recon_coords_head.T).T

        reference_brain = coreg_filenames["smri_file"]

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = reference_brain.replace(".nii.gz", "_{}mm.nii.gz".format(spatial_resolution))

    else:
        ValueError("Invalid out_space, should be mni or mri or scaledmri")

    # ---------------------------------------------------------------------
    # Get coordinates from reference brain at resolution spatial_resolution

    # Create std brain of the required resolution
    rhino_utils.system_call("flirt -in {} -ref {} -out {} -applyisoxfm {}".format(reference_brain, reference_brain, reference_brain_resampled, spatial_resolution))

    coords_out, vals = rhino_utils.niimask2mmpointcloud(reference_brain_resampled)

    # --------------------------------------------------------------
    # For each mni_coords_out find nearest coord in recon_coords_out

    recon_timeseries_out = np.zeros(np.insert(recon_timeseries.shape[1:], 0, coords_out.shape[1]))

    recon_indices = np.zeros([coords_out.shape[1]])

    for cc in range(coords_out.shape[1]):
        recon_index, dist = rhino_utils._closest_node(coords_out[:, cc], recon_coords_out)

        if dist < spatial_resolution:
            recon_timeseries_out[cc, :] = recon_timeseries[recon_index, ...]
            recon_indices[cc] = recon_index

    return recon_timeseries_out, reference_brain_resampled, coords_out, recon_indices


def transform_leadfield(
    subjects_dir,
    subject,
    leadfield,
    spatial_resolution=None,
    reference_brain="mni",
    verbose = None,
):
    """Spatially resamples a (nsensors x ndipoles) array of lead fields (in head/polhemus space) to dipoles on the brain grid of the specified reference brain.

    Parameters
    ----------
    subjects_dir : string
        Directory to find subject directories in.
    subject : string
        Subject name.
    leadfield : numpy.ndarray
        (nsensors, ndipoles) containing the lead field of each dipole (in head (polhemus) space). Assumes that the dipoles are the same (and in the same order)
        as those in the forward model, rhino_files['fwd_model']. Typically derive from the VolSourceEstimate's output by MNE source recon methods,
        e.g. mne.beamformer.apply_lcmv, obtained using a forward model generated by RHINO.
    spatial_resolution : int
        Resolution to use for the reference brain in mm (must be an integer, or will be cast to nearest int). If None, then the gridstep used in rhino_files['fwd_model'] is used.
    reference_brain : string
        'mni' indicates that the reference_brain is the stdbrain in MNI space.
        'mri' indicates that the reference_brain is the subject's sMRI in the scaled native/mri space.
        'unscaled_mri' indicates that the reference_brain is the subject's sMRI in unscaled native/mri space.
        Note that Scaled/unscaled relates to the allow_smri_scaling option in coreg. If allow_scaling was False, then the unscaled MRI will be the same as the scaled MRI.

    Returns
    -------
    leadfield_out : numpy.ndarray
        (nsensors, ndipoles) np.array of lead fields resampled on the reference brain grid.
    """

    rhino_files = rhino_utils.get_rhino_files(subjects_dir, subject)
    surfaces_filenames = rhino_files["surf"]
    coreg_filenames = rhino_files["coreg"]

    # -------------------------------------------------------------------------------------
    # Get hold of coords of points reconstructed to.
    #
    # Note, MNE forward model is done in head space in metres. RHINO does everything in mm.
    fwd = read_forward_solution(rhino_files["fwd_model"], verbose = verbose)
    vs = fwd["src"][0]
    recon_coords_head = vs["rr"][vs["vertno"]] * 1000  # in mm

    # ----------------------------
    if spatial_resolution is None:
        # Estimate gridstep from forward model
        rr = fwd["src"][0]["rr"]

        store = []
        for ii in range(rr.shape[0]):
            store.append(np.sqrt(np.sum(np.square(rr[ii, :] - rr[0, :]))))
        store = np.asarray(store)
        spatial_resolution = int(np.round(np.min(store[np.where(store > 0)]) * 1000))

    spatial_resolution = int(spatial_resolution)

    if reference_brain == "mni":
        # Reference is mni stdbrain

        # Convert recon_coords_head from head to mni space, head_mri_t_file xform is to unscaled MRI
        head_mri_t = rhino_utils.read_trans(coreg_filenames["head_mri_t_file"])
        recon_coords_mri = rhino_utils.xform_points(head_mri_t["trans"], recon_coords_head.T).T

        # mni_mri_t_file xform is to unscaled MRI
        mni_mri_t = rhino_utils.read_trans(surfaces_filenames["mni_mri_t_file"])
        recon_coords_out = rhino_utils.xform_points(np.linalg.inv(mni_mri_t["trans"]), recon_coords_mri.T).T

        reference_brain = os.environ["FSLDIR"] + "/data/standard/MNI152_T1_1mm_brain.nii.gz"

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = op.join(coreg_filenames["basedir"], "MNI152_T1_{}mm_brain.nii.gz".format(spatial_resolution))

    elif reference_brain == "unscaled_mri":
        # Reference is unscaled smri

        # Convert recon_coords_head from head to mri space
        head_mri_t = rhino_utils.read_trans(coreg_filenames["head_mri_t_file"])
        recon_coords_out = rhino_utils.xform_points(head_mri_t["trans"], recon_coords_head.T).T

        reference_brain = surfaces_filenames["smri_file"]

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = reference_brain.replace(".nii.gz", "_{}mm.nii.gz".format(spatial_resolution))

    elif reference_brain == "mri":
        # Reference is scaled smri

        # Convert recon_coords_head from head to mri space
        head_scaledmri_t = rhino_utils.read_trans(coreg_filenames["head_scaledmri_t_file"])
        recon_coords_out = rhino_utils.xform_points(head_scaledmri_t["trans"], recon_coords_head.T).T

        reference_brain = coreg_filenames["smri_file"]

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = reference_brain.replace(".nii.gz", "_{}mm.nii.gz".format(spatial_resolution))

    else:
        ValueError("Invalid out_space, should be mni or mri or scaledmri")

    # ---------------------------------------------------------------------
    # Get coordinates from reference brain at resolution spatial_resolution

    # Create std brain of the required resolution
    rhino_utils.system_call("flirt -in {} -ref {} -out {} -applyisoxfm {}".format( reference_brain, reference_brain, reference_brain_resampled, spatial_resolution))

    coords_out, _ = rhino_utils.niimask2mmpointcloud(reference_brain_resampled)

    # --------------------------------------------------------------
    # For each mni_coords_out find nearest coord in recon_coords_out

    leadfield_out = np.zeros((leadfield.shape[0], coords_out.shape[1]))

    recon_indices = np.zeros([coords_out.shape[1]])

    for cc in range(coords_out.shape[1]):
        recon_index, dist = rhino_utils._closest_node(coords_out[:, cc], recon_coords_out)

        if dist < spatial_resolution:
            leadfield_out[:, cc] = leadfield[:, recon_index]
            recon_indices[cc] = recon_index

    return leadfield_out


@verbose
def _make_lcmv(
    info,
    forward,
    data_cov,
    reg=0.05,
    noise_cov=None,
    label=None,
    pick_ori=None,
    rank="info",
    noise_rank="info",
    weight_norm="unit-noise-gain-invariant",
    reduce_rank=False,
    depth=None,
    inversion="matrix",
    verbose=None,
):
    """Compute LCMV spatial filter.

    Modified version of mne.beamformer._make_lcmv.
    """
    # Code that is different to mne.beamformer.make_lcmv is labelled with MWW

    # Check number of sensor types present in the data and ensure a noise cov
    info = _simplify_info(info)
    noise_cov, _, allow_mismatch = _check_one_ch_type("lcmv", info, forward, data_cov, noise_cov)

    # NOTE: we need this extra picking step (can't just rely on minimum norm's because there can be a mismatch.
    # Should probably add an extra arg to _prepare_beamformer_input at some point (later)
    picks = _check_info_inv(info, forward, data_cov, noise_cov)
    info = pick_info(info, picks)

    data_rank = compute_rank(data_cov, rank=rank, info=info)
    noise_rank = compute_rank(noise_cov, rank=noise_rank, info=info)

    #for key in data_rank:
    #    if (
    #        key not in noise_rank or data_rank[key] != noise_rank[key]
    #    ) and not allow_mismatch:
    #        raise ValueError(
    #            "%s data rank (%s) did not match the noise "
    #            "rank (%s)" % (key, data_rank[key], noise_rank.get(key, None))
    #        )

    # del noise_rank
    rank = data_rank
    mne_logger.info("Making LCMV beamformer with data cov rank {}".format(rank))
    mne_logger.info("Making LCMV beamformer with noise cov rank {}".format(noise_rank))
    del data_rank

    depth = _check_depth(depth, "depth_sparse")
    if inversion == "single":
        depth["combine_xyz"] = False

    is_free_ori, info, proj, vertno, G, whitener, nn, orient_std = _prepare_beamformer_input(
        info, forward, label, pick_ori, noise_cov=noise_cov, rank=noise_rank, pca=False, **depth,
    )

    ch_names = list(info["ch_names"])

    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov._get_square()
    if "estimator" in data_cov:
        del data_cov["estimator"]
    rank_int = sum(rank.values())
    del rank

    # Compute spatial filter
    n_orient = 3 if is_free_ori else 1
    W, max_power_ori = _compute_beamformer(
        G,
        Cm,
        reg,
        n_orient,
        weight_norm,
        pick_ori,
        reduce_rank,
        rank_int,
        inversion=inversion,
        nn=nn,
        orient_std=orient_std,
        whitener=whitener,
    )

    # Get src type to store with filters for _make_stc
    src_type = _get_src_type(forward["src"], vertno)

    # Get subject to store with filters
    subject_from = _subject_from_forward(forward)

    # Is the computed beamformer a scalar or vector beamformer?
    is_free_ori = is_free_ori if pick_ori in [None, "vector"] else False
    is_ssp = bool(info["projs"])

    filters = Beamformer(
        kind="LCMV",
        weights=W,
        data_cov=data_cov,
        noise_cov=noise_cov,
        whitener=whitener,
        weight_norm=weight_norm,
        pick_ori=pick_ori,
        ch_names=ch_names,
        proj=proj,
        is_ssp=is_ssp,
        vertices=vertno,
        is_free_ori=is_free_ori,
        n_sources=forward["nsource"],
        src_type=src_type,
        source_nn=forward["source_nn"].copy(),
        subject=subject_from,
        rank=rank_int,
        max_power_ori=max_power_ori,
        inversion=inversion,
    )

    return filters


def _compute_beamformer(G, Cm, reg, n_orient, weight_norm, pick_ori, reduce_rank, rank, inversion, nn, orient_std, whitener):
    """Compute a spatial beamformer filter (LCMV or DICS).

    For more detailed information on the parameters, see the docstrings of `make_lcmv` and `make_dics`.

    Modified version of mne.beamformer._compute_beamformer.

    Parameters
    ----------
    G : (n_dipoles, n_channels) numpy.ndarray
        The leadfield.
    Cm : (n_channels, n_channels) numpy.ndarray
        The data covariance matrix.
    reg : float
        Regularization parameter.
    n_orient : int
        Number of dipole orientations defined at each source point
    weight_norm : None | 'unit-noise-gain' | 'nai'
        The weight normalization scheme to use.
    pick_ori : None | 'normal' | 'max-power' | max-power-pre-weight-norm
        The source orientation to compute the beamformer in.
    reduce_rank : bool
        Whether to reduce the rank by one during computation of the filter.
    rank : dict | None | 'full' | 'info'
        See compute_rank.
    inversion : 'matrix' | 'single'
        The inversion scheme to compute the weights.
    nn : (n_dipoles, 3) numpy.ndarray
        The source normals.
    orient_std : (n_dipoles,) numpy.ndarray
        The std of the orientation prior used in weighting the lead fields.
    whitener : (n_channels, n_channels) numpy.ndarray
        The whitener.

    Returns
    -------
    W : (n_dipoles, n_channels) numpy.ndarray
        The beamformer filter weights.
    """
    # Lines changes are marked with MWW

    _check_option("weight_norm", weight_norm, ["unit-noise-gain-invariant", "unit-noise-gain", "nai", None])

    # Whiten the data covariance
    Cm = whitener @ Cm @ whitener.T.conj()

    # Restore to properly Hermitian as large whitening coefs can have bad rounding error
    Cm[:] = (Cm + Cm.T.conj()) / 2.0

    assert Cm.shape == (G.shape[0],) * 2

    s, _ = np.linalg.eigh(Cm)
    if not (s >= -s.max() * 1e-7).all():
        # This shouldn't ever happen, but just in case
        warn("data covariance does not appear to be positive semidefinite, results will likely be incorrect")

    # Tikhonov regularization using reg parameter to control for trade-off between spatial resolution and noise sensitivity
    # eq. 25 in Gross and Ioannides, 1999 Phys. Med. Biol. 44 2081
    Cm_inv, loading_factor, rank = _reg_pinv(Cm, reg, rank)

    assert orient_std.shape == (G.shape[1],)

    n_sources = G.shape[1] // n_orient

    assert nn.shape == (n_sources, 3)

    mne_logger.info("Computing beamformer filters for %d source%s" % (n_sources, _pl(n_sources)))
    n_channels = G.shape[0]

    assert n_orient in (3, 1)

    Gk = np.reshape(G.T, (n_sources, n_orient, n_channels)).transpose(0, 2, 1)

    assert Gk.shape == (n_sources, n_channels, n_orient)

    sk = np.reshape(orient_std, (n_sources, n_orient))

    del G, orient_std

    pinv_kwargs = dict()
    if check_version("numpy", "1.17"):
        pinv_kwargs["hermitian"] = True

    _check_option("reduce_rank", reduce_rank, (True, False))

    # inversion of the denominator
    _check_option("inversion", inversion, ("matrix", "single"))

    if inversion == "single" and n_orient > 1 and pick_ori == "vector" and weight_norm == "unit-noise-gain-invariant":
        raise ValueError('Cannot use pick_ori="vector" with inversion="single" and weight_norm="unit-noise-gain-invariant"')

    if reduce_rank and inversion == "single":
        raise ValueError(
            'reduce_rank cannot be used with inversion="single"; consider using inversion="matrix" if you have a rank-deficient forward model '
            '(i.e., from a sphere model with MEG channels), otherwise consider using reduce_rank=False'
        )
    if n_orient > 1:
        _, Gk_s, _ = np.linalg.svd(Gk, full_matrices=False)
        assert Gk_s.shape == (n_sources, n_orient)
        if not reduce_rank and (Gk_s[:, 0] > 1e6 * Gk_s[:, 2]).any():
            raise ValueError("Singular matrix detected when estimating spatial filters. Consider reducing the rank of the forward operator by using reduce_rank=True.")
        del Gk_s

    # --------------------------------
    # 1. Reduce rank of the lead field

    if reduce_rank:
        Gk = _reduce_leadfield_rank(Gk)

    def _compute_bf_terms(Gk, Cm_inv):
        bf_numer = np.matmul(Gk.swapaxes(-2, -1).conj(), Cm_inv)
        bf_denom = np.matmul(bf_numer, Gk)
        return bf_numer, bf_denom

    # ----------------------------------------------------------
    # 2. Reorient lead field in direction of max power or normal

    if pick_ori == "max-power" or pick_ori == "max-power-pre-weight-norm":
        assert n_orient == 3
        _, bf_denom = _compute_bf_terms(Gk, Cm_inv)

        if pick_ori == "max-power":
            if weight_norm is None:
                ori_numer = np.eye(n_orient)[np.newaxis]
                ori_denom = bf_denom
            else:
                # Compute power, cf Sekihara & Nagarajan 2008, eq. 4.47
                ori_numer = bf_denom

                # Cm_inv should be Hermitian so no need for .T.conj()
                ori_denom = np.matmul(
                    np.matmul(Gk.swapaxes(-2, -1).conj(), Cm_inv @ Cm_inv), Gk
                )

            ori_denom_inv = _sym_inv_sm(ori_denom, reduce_rank, inversion, sk)
            ori_pick = np.matmul(ori_denom_inv, ori_numer)

        # MWW
        else:
            # pick_ori == 'max-power-pre-weight-norm':

            # Compute power
            #
            # See eq 5 in Brookes et al, Optimising experimental design for MEG beamformer imaging, Neuroimage 2008
            # This optimises the orientation by maximising the power BEFORE any weight normalisation is performed
            ori_pick = _sym_inv_sm(bf_denom, reduce_rank, inversion, sk)

        assert ori_pick.shape == (n_sources, n_orient, n_orient)

        # Pick eigenvector that corresponds to maximum eigenvalue
        eig_vals, eig_vecs = np.linalg.eig(ori_pick.real)  # not Hermitian!

        # Sort eigenvectors by eigenvalues for picking
        order = np.argsort(np.abs(eig_vals), axis=-1)
        #eig_vals = np.take_along_axis(eig_vals, order, axis=-1)
        max_power_ori = eig_vecs[np.arange(len(eig_vecs)), :, order[:, -1]]

        assert max_power_ori.shape == (n_sources, n_orient)

        # Set the (otherwise arbitrary) sign to match the normal
        signs = np.sign(np.sum(max_power_ori * nn, axis=1, keepdims=True))
        signs[signs == 0] = 1.0
        max_power_ori *= signs

        # Compute the lead field for the optimal orientation, and adjust numer/denom
        Gk = np.matmul(Gk, max_power_ori[..., np.newaxis])
        n_orient = 1

    else:
        max_power_ori = None
        if pick_ori == "normal":
            Gk = Gk[..., 2:3]
            n_orient = 1

    # ----------------------------------------------------------------------
    # 3. Compute numerator and denominator of beamformer formula (unit-gain)

    bf_numer, bf_denom = _compute_bf_terms(Gk, Cm_inv)

    assert bf_denom.shape == (n_sources,) + (n_orient,) * 2
    assert bf_numer.shape == (n_sources, n_orient, n_channels)

    del Gk  # lead field has been adjusted and should not be used anymore

    # ----------------------------------------------------------------------
    # 4. Invert the denominator

    # Here W is W_ug, i.e.: G.T @ Cm_inv / (G.T @ Cm_inv @ G)
    bf_denom_inv = _sym_inv_sm(bf_denom, reduce_rank, inversion, sk)

    assert bf_denom_inv.shape == (n_sources, n_orient, n_orient)

    W = np.matmul(bf_denom_inv, bf_numer)

    assert W.shape == (n_sources, n_orient, n_channels)

    del bf_denom_inv, sk

    # ----------------------------------------------------------------
    # 5. Re-scale filter weights according to the selected weight_norm

    # Weight normalization is done by computing, for each source:
    #
    #     W_ung = W_ug / sqrt(W_ug @ W_ug.T)
    #
    # with W_ung referring to the unit-noise-gain (weight normalized) filter and W_ug referring to the above-calculated unit-gain filter stored in W.

    if weight_norm is not None:
        # Three different ways to calculate the normalization factors here. Only matters when in vector mode,
        # as otherwise n_orient == 1 and they are all equivalent. Sekihara 2008 says to use
        #
        # In MNE < 0.21, we just used the Frobenius matrix norm:
        #
        #    noise_norm = np.linalg.norm(W, axis=(1, 2), keepdims=True)
        #    assert noise_norm.shape == (n_sources, 1, 1)
        #    W /= noise_norm
        #
        # Sekihara 2008 says to use sqrt(diag(W_ug @ W_ug.T)), which is not rotation invariant:
        if weight_norm in ("unit-noise-gain", "nai"):
            noise_norm = np.matmul(W, W.swapaxes(-2, -1).conj()).real
            noise_norm = np.reshape(noise_norm, (n_sources, -1, 1))[:, :: n_orient + 1] # np.diag operation over last two axes
            np.sqrt(noise_norm, out=noise_norm)
            noise_norm[noise_norm == 0] = np.inf
            assert noise_norm.shape == (n_sources, n_orient, 1)
            W /= noise_norm
        else:
            assert weight_norm == "unit-noise-gain-invariant"
            # Here we use sqrtm. The shortcut:
            #
            #    use = W
            #
            # ... does not match the direct route (it is rotated!), so we'll use the direct one to match FieldTrip:
            use = bf_numer
            inner = np.matmul(use, use.swapaxes(-2, -1).conj())
            W = np.matmul(_sym_mat_pow(inner, -0.5), use)
            noise_norm = 1.0

        if weight_norm == "nai":
            # Estimate noise level based on covariance matrix, taking the first eigenvalue that falls outside the signal subspace
            # or the loading factor used during regularization, whichever is largest.
            if rank > len(Cm):
                # Covariance matrix is full rank, no noise subspace!
                # Use the loading factor as noise ceiling.
                if loading_factor == 0:
                    raise RuntimeError(
                        "Cannot compute noise subspace with a full-rank covariance matrix and no regularization. "
                        "Try manually specifying the rank of the covariance matrix or using regularization."
                    )
                noise = loading_factor
            else:
                noise, _ = np.linalg.eigh(Cm)
                noise = noise[-rank]
                noise = max(noise, loading_factor)
            W /= np.sqrt(noise)

    W = W.reshape(n_sources * n_orient, n_channels)
    mne_logger.info("Filter computation complete")
    return W, max_power_ori


def _prepare_beamformer_input(
    info,
    forward,
    label=None,
    pick_ori=None,
    noise_cov=None,
    rank=None,
    pca=False,
    loose=None,
    combine_xyz="fro",
    exp=None,
    limit=None,
    allow_fixed_depth=True,
    limit_depth_chs=False,
):
    """Input preparation common for LCMV, DICS, and RAP-MUSIC.

    Modified version of mne.beamformer._prepare_beamformer_input.
    """
    # Lines marked MWW are where code has been changed.

    # MWW
    # _check_option('pick_ori', pick_ori, ('normal', 'max-power', 'vector', None))
    _check_option("pick_ori", pick_ori, ("normal", "max-power", "vector", "max-power-pre-weight-norm", None))

    # MWW
    # Restrict forward solution to selected vertices
    #if label is not None:
    #    _, src_sel = label_src_vertno_sel(label, forward["src"])
    #    forward = _restrict_forward_to_src_sel(forward, src_sel)

    if loose is None:
        loose = 0.0 if is_fixed_orient(forward) else 1.0

    # MWW
    #if noise_cov is None:
    #    noise_cov = make_ad_hoc_cov(info, std=1.0)

    forward, info_picked, gain, _, orient_prior, _, trace_GRGT, noise_cov, whitener = _prepare_forward(
        forward,
        info,
        noise_cov,
        "auto",
        loose,
        rank=rank,
        pca=pca,
        use_cps=True,
        exp=exp,
        limit_depth_chs=limit_depth_chs,
        combine_xyz=combine_xyz,
        limit=limit,
        allow_fixed_depth=allow_fixed_depth,
    )
    is_free_ori = not is_fixed_orient(forward)  # could have been changed
    nn = forward["source_nn"]
    if is_free_ori:  # take Z coordinate
        nn = nn[2::3]
    nn = nn.copy()
    vertno = _get_vertno(forward["src"])
    if forward["surf_ori"]:
        nn[...] = [0, 0, 1]  # align to local +Z coordinate
    if pick_ori is not None and not is_free_ori:
        raise ValueError("Normal or max-power orientation (got {}) can only be picked when a forward operator with free orientation is used.".format(pick_ori))
    if pick_ori == "normal" and not forward["surf_ori"]:
        raise ValueError("Normal orientation can only be picked when a forward operator oriented in surface coordinates is used.")
    _check_src_normal(pick_ori, forward["src"])
    del forward, info

    # Undo the scaling that MNE prefers
    scale = np.sqrt((noise_cov["eig"] > 0).sum() / trace_GRGT)
    gain /= scale
    if orient_prior is not None:
        orient_std = np.sqrt(orient_prior)
    else:
        orient_std = np.ones(gain.shape[1])

    # Get the projector
    proj, _, _ = make_projector(info_picked["projs"], info_picked["ch_names"])

    return is_free_ori, info_picked, proj, vertno, gain, whitener, nn, orient_std


def voxel_timeseries(
    subjects_dir,
    subject,
    preproc_file,
    chantypes,
    freq_range=None,
    spatial_resolution=None,
    reference_brain="mni",
    reject_by_annotation=None,
):
    """Get the voxel time series of beamformed data.

    Parameters
    ----------
    subjects_dir : string
        Directory to find subject directories in.
    subject : string
        Subject name.
    preproc_file : str
        Path to the preprocessed fif file.
    chantypes : list of str
        Channel types to use in beamforming.
    freq_range : list
        Lower and upper band to bandpass filter before beamforming. If None, no filtering is done.
    spatial_resolution : int
        Resolution for beamforming to use for the reference brain in mm (must be an integer, or will be cast to nearest int).
        If None, then the gridstep used in rhino_files['fwd_model'] is used.
    reference_brain : string
        'mni' indicates that the reference_brain is the stdbrain in MNI space.
        'mri' indicates that the reference_brain is the subject's sMRI in the scaled native/mri space.
        'unscaled_mri' indicates that the reference_brain is the subject's sMRI in unscaled native/mri space.
        Note that Scaled/unscaled relates to the allow_smri_scaling option in coreg. If allow_scaling was False, then the unscaled MRI will be the same as the scaled MRI.
    reject_by_annotation : str
        Argument passed to .get_data() if the preproc file contains an MNE Raw object.

    Returns
    -------
    voxel_data : np.ndarray
        Voxel time series in (voxels, time) format.
    voxel_coords : np.ndarray
        Voxel coordinates in MNI space.
    """

    # Load sensor data
    if "raw.fif" in preproc_file:
        # Load preprocessed data
        data = mne.io.read_raw_fif(preproc_file, preload=True)
    elif "epo.fif" in preproc_file:
        # Load epoched data
        data = mne.read_epochs(preproc_file, preload=True)
    else:
        raise ValueError("fif file must end in 'raw.fif' or 'epo.fif'.")

    log_or_print(f"using chantypes: {chantypes}")
    if isinstance(chantypes, str):
        chantypes = [chantypes]
    data.pick(chantypes)

    # Bandpass filter
    if freq_range is not None:
        log_or_print(f"bandpass filtering: {freq_range}")
        data = data.filter(l_freq=freq_range[0], h_freq=freq_range[1], method="iir", iir_params={"order": 5, "ftype": "butter"})

    # Load the beamformer and apply
    filters = load_lcmv(subjects_dir, subject)
    bf_data = apply_lcmv(data, filters, reject_by_annotation)

    # Transform to MNI space
    if "epo.fif" in preproc_file:
        bf_data = np.transpose([bf.data for bf in bf_data], axes=[1, 2, 0])
    else:
        bf_data = bf_data.data
    voxel_timeseries, _, voxel_coords, _ = transform_recon_timeseries(
        subjects_dir=subjects_dir,
        subject=subject,
        recon_timeseries=bf_data,
        spatial_resolution=spatial_resolution,
        reference_brain=reference_brain,
    )

    return voxel_timeseries, voxel_coords
