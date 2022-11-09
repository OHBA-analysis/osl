#!/usr/bin/env python

"""Functions and classes to handle parcellation.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import os.path as op
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy.spatial import KDTree
from nilearn.plotting import plot_markers
from mpl_toolkits.axes_grid1 import make_axes_locatable

import osl.source_recon.rhino.utils as rhino_utils
from osl.utils import soft_import
from osl.utils.logger import log_or_print


class Parcellation:
    def __init__(self, file):
        if isinstance(file, Parcellation):
            self.__dict__.update(file.__dict__)
            return
        self.file = find_file(file)
        self.parcellation = nib.load(self.file)
        self.dims = self.parcellation.shape[:3]
        self.n_parcels = self.parcellation.shape[3]
        self.parcel_timeseries = None

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.file)})"

    def data(self):
        return self.parcellation.get_fdata()

    def nonzero(self):
        return [np.nonzero(self.data()[..., i]) for i in range(self.n_parcels)]

    def nonzero_coords(self):
        return [
            nib.affines.apply_affine(self.parcellation.affine, np.array(nonzero).T)
            for nonzero in self.nonzero()
        ]

    def weights(self):
        return [
            self.data()[..., i][nonzero] for i, nonzero in enumerate(self.nonzero())
        ]

    def roi_centers(self):
        return np.array(
            [
                np.average(c, weights=w, axis=0)
                for c, w in zip(self.nonzero_coords(), self.weights())
            ]
        )

    def plot(self, **kwargs):
        return plot_parcellation(self, **kwargs)

    def parcellate(
        self,
        voxel_timeseries,
        voxel_coords,
        method="spatial_basis",
        working_dir=None,
        logger=None,
    ):
        """Parcellate voxel time series.

        Parameters
        ----------
        voxel_timeseries : numpy.ndarray
            nvoxels x ntpts, or nvoxels x ntpts x ntrials
            Data to be parcellated. Data is assumed to be in same space as the
            parcellation (e.g. typically corresponds to the output from
            rhino.resample_recon_ts).
        voxel_coords : numpy.ndarray
            (nvoxels x 3) coordinates of voxel_timeseries in mm in same space as
            parcellation (e.g. typically corresponds to the output from
            rhino.resample_recon_ts).
        method : str
            'pca' - take 1st PC in each parcel
            'spatial_basis' - The parcel time-course for each spatial map is the
            1st PC from all voxels, weighted by the spatial map. If the parcellation
            is unweighted and non-overlapping, 'spatialBasis' will give the same
            result as 'PCA' except with a different normalization.
        working_dir : str
            Dir to put temp file in. If None, attempt to use same dir as passed in
            parcellation.
        logger : logging.getLogger
            Logger.

        Returns
        -------
        parcel_timeseries : dict
            Containing:
            "data": numpy.ndarray
                nparcels x ntpts, or nparcels x ntpts x ntrials, parcellated data
            "voxel_coords": numpy.ndarray
                Passed in (nvoxels x 3) coordinates of voxel_timeseries
            "voxel_weightings": numpy.ndarray
                nvoxels x nparcels
                Voxel weightings for each parcel, corresponds to
                parcel_data = voxel_weightings.T * voxel_data
            "voxel_assignments": bool numpy.ndarray
                nvoxels x nparcels
                Boolean assignments indicating for each voxel the winner takes all
                parcel it belongs to
        """
        parcellation_asmatrix = _resample_parcellation(
            self, voxel_coords, working_dir, logger=logger
        )
        data, voxel_weightings, voxel_assignments = _get_parcel_timeseries(
            voxel_timeseries, parcellation_asmatrix, method=method
        )

        self.parcel_timeseries = {
            "data": data,
            "voxel_coords": voxel_coords,
            "voxel_weightings": voxel_weightings,
            "voxel_assignments": voxel_assignments,
        }

    def symmetric_orthogonalise(self, maintain_magnitudes=False, compute_weights=False):
        self.parcel_timeseries["data"] = symmetric_orthogonalise(
            self.parcel_timeseries["data"],
            maintain_magnitudes=maintain_magnitudes,
            compute_weights=compute_weights,
        )

    def nii(
        self,
        parcel_timeseries_data,
        method="assignments",
        out_nii_fname=None,
        working_dir=None,
        times=None,
    ):
        """Outputs parcel_timeseries_data as a niftii file using the parcellation

        Parameters
        ----------
        parcel_timeseries_data : numpy.ndarray
            Needs to have same number of parcels as the parcellation
            Needs to be nparcels x ntpts
        method : str
            Method used to allocate values to voxels given the parcel values,
            use either:
                "weights" - Voxel weightings for each parcel
                "assignments" - Boolean assignments indicating the winner-takes-all
                parcel each voxel belongs to
        working_dir : str
            Dir name to put files in
        out_nii_fname : str
            Output nii file name, will be output at spatial resolution of
            parcel_timeseries['voxel_coords']. If None then will generate a name
        times : (ntpts, ) numpy.ndarray
            Times points in seconds.
            Will assume that time points are regularly spaced.
            Used to set up 4D nii files correctly.

        Returns
        -------
        out_nii_fname : str
            Output nii file name, will be output at spatial resolution of
            parcel_timeseries['voxel_coords']

        """

        if self.parcel_timeseries is None:
            raise ValueError(
                "You need to have run Parcellation.parcellate() or "
                + "Parcellation.load_parcel_timeseries() prior to calling"
                + "Parcellation.nii()."
            )

        result = _parcel_timeseries2nii(
            self,
            parcel_timeseries_data,
            method=method,
            out_nii_fname=out_nii_fname,
            working_dir=working_dir,
            times=times,
        )

        return result

    def load_parcel_timeseries(self, fname):
        self.parcel_timeseries = _load_parcel_timeseries(fname)
        return self.parcel_timeseries

    def save_parcel_timeseries(self, fname):
        _save_parcel_timeseries(self.parcel_timeseries, fname)


def find_file(filename):
    if not op.exists(filename):
        files_dir = str(Path(__file__).parent) + "/files/"
        if op.exists(files_dir + filename):
            filename = files_dir + filename
        else:
            raise FileNotFoundError(filename)
    return filename


def plot_parcellation(parcellation, **kwargs):
    parcellation = Parcellation(parcellation)
    return plot_markers(
        np.zeros(parcellation.n_parcels),
        parcellation.roi_centers(),
        colorbar=False,
        node_cmap="binary_r",
        **kwargs,
    )


def plot_correlation(parc_ts, filename, logger=None):
    corr = np.corrcoef(parc_ts)
    np.fill_diagonal(corr, 0)
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(corr)
    ax.set_title("Correlation")
    ax.set_xlabel("Parcel")
    ax.set_ylabel("Parcel")
    fig.colorbar(im, cax=cax, orientation='vertical')
    log_or_print(f"saving {filename}", logger)
    fig.savefig(filename)
    plt.close(fig)


def _resample_parcellation(
    parcellation, voxel_coords, working_dir=None, logger=None,
):
    """Resample parcellation so that its voxel coords correspond (using
    nearest neighbour) to passed in voxel_coords. Passed in voxel_coords
    and parcellation must be in the same space, e.g. MNI.

    Used to make sure that the parcellation's voxel coords are the same
    as the voxel coords for some timeseries data, before calling
    get_parcel_timeseries.

    Parameters
    ----------
    parcellation : parcellation.Parcellation
        In same space as voxel_coords.
    voxel_coords :
        (nvoxels x 3) coordinates in mm in same space as parcellation.
    working_dir : str
        Dir to put temp file in. If None, attempt to use same dir as passed
        in parcellation.
    logger : logging.getLogger
        Logger.

    Returns
    -------
    parcellation_asmatrix : numpy.ndarray
        (nvoxels x nparcels) resampled parcellation
    """
    gridstep = int(rhino_utils.get_gridstep(voxel_coords.T) / 1000)
    log_or_print(f"gridstep = {gridstep} mm", logger)

    pth, parcellation_name = op.split(op.splitext(op.splitext(parcellation.file)[0])[0])

    if working_dir is None:
        working_dir = pth

    parcellation_resampled = op.join(
        working_dir, parcellation_name + "_{}mm.nii.gz".format(gridstep)
    )

    # create std brain of the required resolution
    os.system(
        "flirt -in {} -ref {} -out {} -applyisoxfm {}".format(
            parcellation.file, parcellation.file, parcellation_resampled, gridstep
        )
    )

    nparcels = nib.load(parcellation_resampled).get_fdata().shape[3]

    # parcellation_asmatrix will be the parcels mapped onto the same dipole grid
    # as voxel_coords
    parcellation_asmatrix = np.zeros((voxel_coords.shape[1], nparcels))

    for parcel_index in range(nparcels):
        parcellation_coords, parcellation_vals = rhino_utils.niimask2mmpointcloud(
            parcellation_resampled, parcel_index
        )

        kdtree = KDTree(parcellation_coords.T)

        # Find each voxel_coords best matching parcellation_coords and assign
        # the corresponding parcel value to
        for ind in range(voxel_coords.shape[1]):
            distance, index = kdtree.query(voxel_coords[:, ind])

            # Exclude from parcel any voxel_coords that are further than gridstep
            # away from the best matching parcellation_coords
            if distance < gridstep:
                parcellation_asmatrix[ind, parcel_index] = parcellation_vals[index]

    return parcellation_asmatrix


def _get_parcel_timeseries(
    voxel_timeseries, parcellation_asmatrix, method="spatial_basis"
):
    """Calculate parcel timeseries

    Parameters
    ----------
    parcellation_asmatrix: numpy.ndarray
        nvoxels x nparcels
        And is assumed to be on the same grid as voxel_timeseries
    voxel_timeseries : numpy.ndarray
        nvoxels x ntpts, or nvoxels x ntpts x ntrials
        And is assumed to be on the same grid as parcellation
        (typically output by rhino.resample_recon_ts)
    method : str
        'pca'   - take 1st PC of voxels
        'spatial_basis' - The parcel time-course for each spatial map is the
        1st PC from all voxels, weighted by the spatial map. If the parcellation
        is unweighted and non-overlapping, 'spatialBasis' will give the same
        result as 'PCA' except with a different normalization

    Returns
    -------
    parcel_timeseries : numpy.ndarray
        nparcels x ntpts, or nparcels x ntpts x ntrials
    voxel_weightings : numpy.ndarray
        nvoxels x nparcels
        Voxel weightings for each parcel to compute parcel_timeseries from
        voxel_timeseries
    voxel_assignments : bool numpy.ndarray
        nvoxels x nparcels
        Boolean assignments indicating for each voxel the winner takes all
        parcel it belongs to
    """

    if parcellation_asmatrix.shape[0] != voxel_timeseries.shape[0]:
        Exception(
            "Parcellation has {} voxels, but data has {}".format(
                parcellation_asmatrix.shape[0], voxel_timeseries.shape[0]
            )
        )

    if len(voxel_timeseries.shape) == 2:
        # add dim for trials:
        voxel_timeseries = np.expand_dims(voxel_timeseries, axis=2)
        added_dim = True
    else:
        added_dim = False

    nparcels = parcellation_asmatrix.shape[1]
    ntpts = voxel_timeseries.shape[1]
    ntrials = voxel_timeseries.shape[2]

    # combine the trials and time dimensions together,
    # we will re-separate them after the parcel timeseries are computed
    voxel_timeseries_reshaped = np.reshape(
        voxel_timeseries, (voxel_timeseries.shape[0], ntpts * ntrials)
    )
    parcel_timeseries_reshaped = np.zeros((nparcels, ntpts * ntrials))

    voxel_weightings = np.zeros(parcellation_asmatrix.shape)

    if method == "spatial_basis":
        # estimate temporal-STD of data for normalisation
        temporal_std = np.maximum(
            np.std(voxel_timeseries_reshaped, axis=1), np.finfo(float).eps
        )

        for pp in range(nparcels):
            # scale group maps so all have a positive peak of height 1
            # in case there is a very noisy outlier, choose the sign from the
            # top 5% of magnitudes
            thresh = np.percentile(np.abs(parcellation_asmatrix[:, pp]), 95)
            mapsign = np.sign(
                np.mean(
                    parcellation_asmatrix[parcellation_asmatrix[:, pp] > thresh, pp]
                )
            )
            scaled_parcellation = (
                mapsign
                * parcellation_asmatrix[:, pp]
                / np.max(np.abs(parcellation_asmatrix[:, pp]))
            )

            # Weight all voxels by the spatial map in question
            # Apply the mask first then weight, to reduce memory use
            weighted_ts = voxel_timeseries_reshaped[scaled_parcellation > 0, :]
            weighted_ts = np.multiply(
                weighted_ts,
                np.reshape(scaled_parcellation[scaled_parcellation > 0], [-1, 1]),
            )

            # Perform svd and take scores of 1st PC as the node time-series
            # U is nVoxels by nComponents - the basis transformation
            # S*V holds nComponents by time sets of PCA scores - the
            # timeseries data in the new basis
            d, U = scipy.sparse.linalg.eigs(weighted_ts @ weighted_ts.T, k=1)
            U = np.real(U)
            d = np.real(d)
            S = np.sqrt(np.abs(np.real(d)))
            V = weighted_ts.T @ U / S
            pca_scores = S @ V.T

            # 0.5 is a decent arbitrary threshold used in fslnets after playing
            # with various maps
            this_mask = scaled_parcellation[scaled_parcellation > 0] > 0.5

            if np.any(this_mask):  # the mask is non-zero
                # U is the basis by which voxels in the mask are weighted
                # to form the scores of the 1st PC
                relative_weighting = np.abs(U[this_mask]) / np.sum(np.abs(U[this_mask]))
                ts_sign = np.sign(np.mean(U[this_mask]))
                ts_scale = np.dot(
                    np.reshape(relative_weighting, [-1]),
                    temporal_std[scaled_parcellation > 0][this_mask],
                )

                node_ts = (
                    ts_sign
                    * (ts_scale / np.maximum(np.std(pca_scores), np.finfo(float).eps))
                    * pca_scores
                )

                inds = np.where(scaled_parcellation > 0)[0]
                voxel_weightings[inds, pp] = (
                    ts_sign
                    * ts_scale
                    / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                    * (
                        np.reshape(U, [-1])
                        * scaled_parcellation[scaled_parcellation > 0].T
                    )
                )

            else:
                print(
                    "WARNING: An empty parcel mask was found for parcel {} ".format(pp)
                    + "when calculating its time-courses\n"
                    + "The parcel will have a flat zero time-course.\n"
                    + "Check this does not cause further problems with the analysis.\n"
                )

                node_ts = np.zeros(ntpts * ntrials)
                inds = np.where(scaled_parcellation > 0)[0]
                voxel_weightings[inds, pp] = 0

            parcel_timeseries_reshaped[pp, :] = node_ts

    elif method == "pca":
        print(
            "PCA assumes a binary parcellation.\n"
            "Parcellation will be binarised if it is not already "
            "(any voxels >0 are set to 1, otherwise voxels are set to 0), "
            "i.e. any weightings will be ignored.\n"
        )

        # check that each voxel is only a member of one parcel
        if any(np.sum(parcellation_asmatrix, axis=1) > 1):
            print(
                "WARNING: Each voxel is meant to be a member of at most one parcel, "
                "when using the PCA method.\n"
                "Results may not be sensible"
            )

        # estimate temporal-STD of data for normalisation
        temporal_std = np.maximum(
            np.std(voxel_timeseries_reshaped, axis=1), np.finfo(float).eps
        )

        # perform PCA on each parcel and select 1st PC scores to represent parcel
        for pp in range(nparcels):

            if any(parcellation_asmatrix[:, pp]):  # non-zero
                parcel_data = voxel_timeseries_reshaped[
                    parcellation_asmatrix[:, pp] > 0, :
                ]
                parcel_data = parcel_data - np.reshape(
                    np.mean(parcel_data, axis=1), [-1, 1]
                )

                # Perform svd and take scores of 1st PC as the node time-series
                # U is nVoxels by nComponents - the basis transformation
                # S*V holds nComponents by time sets of PCA scores - the
                # timeseries data in the new basis

                d, U = scipy.sparse.linalg.eigs(parcel_data @ parcel_data.T, k=1)
                U = np.real(U)
                d = np.real(d)
                S = np.sqrt(np.abs(np.real(d)))
                V = parcel_data.T @ U / S
                pca_scores = S @ V.T

                # Restore sign and scaling of parcel time-series
                # U indicates the weight with which each voxel in the
                # parcel contributes to the 1st PC
                relative_weighting = np.abs(U) / np.sum(np.abs(U))
                ts_sign = np.sign(np.mean(U))
                ts_scale = np.dot(
                    np.reshape(relative_weighting, [-1]),
                    temporal_std[parcellation_asmatrix[:, pp] > 0],
                )

                node_ts = (
                    ts_sign
                    * ts_scale
                    / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                ) * pca_scores

                inds = np.where(parcellation_asmatrix[:, pp] > 0)[0]
                voxel_weightings[inds, pp] = (
                    ts_sign
                    * ts_scale
                    / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                    * np.reshape(U, [-1])
                )

            else:
                print(
                    "WARNING: An empty parcel mask was found for parcel {}".format(pp)
                    + " when calculating its time-courses\n"
                    + "The parcel will have a flat zero time-course.\n"
                    + "Check this does not cause further problems with the analysis.\n"
                )

                node_ts = np.zeros(ntpts * ntrials)
                inds = np.where(parcellation_asmatrix[:, pp] > 0)[0]
                voxel_weightings[inds, pp] = 0

            parcel_timeseries_reshaped[pp, :] = node_ts

    else:
        Exception("Invalid method specified")

    # Re-separate the trials and time dimensions
    parcel_timeseries = np.reshape(
        parcel_timeseries_reshaped, (nparcels, ntpts, ntrials)
    )
    if added_dim:
        parcel_timeseries = np.squeeze(parcel_timeseries, axis=2)

    # compute voxel_assignments using winner takes all
    voxel_assignments = np.zeros(voxel_weightings.shape)
    for ivoxel in range(voxel_weightings.shape[0]):
        winning_parcel = np.argmax(voxel_weightings[ivoxel, :])
        voxel_assignments[ivoxel, winning_parcel] = 1

    return parcel_timeseries, voxel_weightings, voxel_assignments


def _parcel_timeseries2nii(
    parcellation,
    parcel_timeseries_data,
    out_nii_fname=None,
    working_dir=None,
    times=None,
    method="assignments",
):
    """Outputs parcel_timeseries_data as a niftii file using passed in parcellation,
    parcellation and parcel_timeseries_data need to have the same number of parcels.

    Parameters
    ----------
    parcellation : parcellation.Parcellation
        Parcellation to use
    parcel_timeseries_data: numpy.ndarray
        Needs to be nparcels x ntpts
    working_dir : str
        Dir name to put files in
    out_nii_fname : str
        Output name to put files in
    times : (ntpts, ) numpy.ndarray
        Times points in seconds.
        Will assume that time points are regularly spaced.
        Used to set nii file up correctly.
    method : str
        "weights" or "assignments"

    Returns
    -------
    out_nii_fname : str
        Output nii file name, will be output at spatial resolution of
        parcel_timeseries['voxel_coords']
    """
    pth, parcellation_name = op.split(op.splitext(op.splitext(parcellation.file)[0])[0])

    if working_dir is None:
        working_dir = pth

    if out_nii_fname is None:
        out_nii_fname = op.join(working_dir, parcellation_name + "_timeseries.nii.gz")

    # compute parcellation_mask_file to be mean over all parcels
    parcellation_mask_file = op.join(working_dir, parcellation_name + "_mask.nii.gz")
    os.system("fslmaths {} -Tmean {}".format(parcellation.file, parcellation_mask_file))

    if len(parcel_timeseries_data.shape) == 1:
        parcel_timeseries_data = np.reshape(
            parcel_timeseries_data, [parcel_timeseries_data.shape[0], 1]
        )

    if parcellation.n_parcels != parcel_timeseries_data.shape[0]:
        raise ValueError(
            "Passed in data are not compatible with passed in parcellation - "
            + "they need to have the same number of parcels.\n"
            + "p.n_parcels = {} \nparcel_timeseries_data.shape[0] = {} \n".format(
                parcellation.n_parcels, parcel_timeseries_data.shape[0]
            )
        )

    # Compute nmaskvoxels x ntpts voxel_data
    if method == "assignments":
        weightings = parcellation.parcel_timeseries["voxel_assignments"]
    elif method == "weights":
        # parcel_timeseries_data = voxel_weightings.T *  voxel_data
        # voxel_weightings were computed by parcellation.parcellate()
        weightings = np.linalg.pinv(
            parcellation.parcel_timeseries["voxel_weightings"].T
        )
    else:
        raise ValueError("Invalid method. Must be assignments or weights.")

    voxel_data = weightings @ parcel_timeseries_data

    # voxel_coords is nmaskvoxels x 3 in mm
    voxel_coords = parcellation.parcel_timeseries["voxel_coords"]

    gridstep = int(rhino_utils.get_gridstep(voxel_coords.T) / 1000)

    # Sample parcellation_mask to the desired resolution
    pth, ref_brain_name = op.split(
        op.splitext(op.splitext(parcellation_mask_file)[0])[0]
    )
    parcellation_mask_resampled = op.join(
        working_dir, ref_brain_name + "_{}mm_brain.nii.gz".format(gridstep)
    )

    # create std brain of the required resolution
    os.system(
        "flirt -in {} -ref {} -out {} -applyisoxfm {}".format(
            parcellation_mask_file,
            parcellation_mask_file,
            parcellation_mask_resampled,
            gridstep,
        )
    )

    parcellation_mask_coords, vals = rhino_utils.niimask2mmpointcloud(
        parcellation_mask_resampled
    )
    parcellation_mask_inds = rhino_utils.niimask2indexpointcloud(
        parcellation_mask_resampled
    )

    vol = nib.load(parcellation_mask_resampled).get_fdata()
    vol = np.zeros(np.append(vol.shape[:3], parcel_timeseries_data.shape[1]))
    kdtree = KDTree(parcellation_mask_coords.T)

    # Find each voxel_coords best matching parcellation_mask_coords
    for ind in range(voxel_coords.shape[1]):
        distance, index = kdtree.query(voxel_coords[:, ind])
        # Exclude from parcel any voxel_coords that are further than gridstep away
        if distance < gridstep:
            vol[
                parcellation_mask_inds[0, index],
                parcellation_mask_inds[1, index],
                parcellation_mask_inds[2, index],
                :,
            ] = voxel_data[ind, :]

    # SAVE AS NIFTI
    vol_nii = nib.Nifti1Image(vol, nib.load(parcellation_mask_resampled).affine)

    vol_nii.header.set_xyzt_units(2)  # mm
    if times is not None:
        vol_nii.header["pixdim"][4] = times[1] - times[0]
        vol_nii.header["toffset"] = 0
        vol_nii.header.set_xyzt_units(2, 8)  # mm and secs

    nib.save(vol_nii, out_nii_fname)

    return out_nii_fname


def symmetric_orthogonalise(
    timeseries, maintain_magnitudes=False, compute_weights=False
):
    """Returns orthonormal matrix L which is closest to A, as measured by the
    Frobenius norm of (L-A). The orthogonal matrix is constructed from a singular
    value decomposition of A.

    If maintain_magnitudes is True, returns the orthogonal matrix L, whose columns
    have the same magnitude as the respective columns of A, and which is closest to
    A, as measured by the Frobenius norm of (L-A)

    Parameters
    ----------
    timeseries : numpy.ndarray
        (nparcels x ntpts) or (nparcels x ntpts x ntrials) data to orthoganlise.
        In the latter case, the ntpts and ntrials dimensions are concatenated.
    maintain_magnitudes : bool
    compute_weights : bool

    Returns
    -------
    ortho_timeseries : numpy.ndarray
        (nparcels x ntpts) or (nparcels x ntpts x ntrials) orthoganalised data
    weights : numpy.ndarray
        (optional output depending on compute_weights flag)
        weighting matrix such that, ortho_timeseries = timeseries * weights

    Reference
    ---------
    Colclough, G. L., Brookes, M., Smith, S. M. and Woolrich, M. W.,
    "A symmetric multivariate leakage correction for MEG connectomes,"
    NeuroImage 117, pp. 439-448 (2015)
    """

    if len(timeseries.shape) == 2:
        # add dim for trials:
        timeseries = np.expand_dims(timeseries, axis=2)
        added_dim = True
    else:
        added_dim = False

    nparcels = timeseries.shape[0]
    ntpts = timeseries.shape[1]
    ntrials = timeseries.shape[2]
    compute_weights = False

    # combine the trials and time dimensions together,
    # we will re-separate them after the parcel timeseries are computed
    timeseries = np.transpose(np.reshape(timeseries, (nparcels, ntpts * ntrials)))

    if maintain_magnitudes:
        D = np.diag(np.sqrt(np.diag(np.transpose(timeseries) @ timeseries)))
        timeseries = timeseries @ D

    [U, S, V] = np.linalg.svd(timeseries, full_matrices=False)

    # we need to check that we have sufficient rank
    tol = max(timeseries.shape) * S[0] * np.finfo(type(timeseries[0, 0])).eps
    r = sum(S > tol)
    full_rank = r >= timeseries.shape[1]

    if full_rank:
        # polar factors of A
        ortho_timeseries = U @ np.conjugate(V)
    else:
        raise ValueError("Not full rank, rank required is {}, but rank is only {}".format(timeseries.shape[1], r))

    if compute_weights:
        # weights are a weighting matrix such that,
        # ortho_timeseries = timeseries * weights
        weights = np.transpose(V) @ np.diag(1.0 / S) @ np.conjugate(V)

    if maintain_magnitudes:
        # scale result
        ortho_timeseries = ortho_timeseries @ D

        if compute_weights:
            # weights are a weighting matrix such that,
            # ortho_timeseries = timeseries * weights
            weights = D @ weights @ D

    # Re-separate the trials and time dimensions
    ortho_timeseries = np.reshape(
        np.transpose(ortho_timeseries), (nparcels, ntpts, ntrials)
    )

    if added_dim:
        ortho_timeseries = np.squeeze(ortho_timeseries, axis=2)

    if compute_weights:
        return ortho_timeseries, weights
    else:
        return ortho_timeseries


def _save_parcel_timeseries(ts, fname):
    # saves passed in dictionary, ts, as a hd5 file
    dd = soft_import("deepdish")
    dd.io.save(fname, ts)


def _load_parcel_timeseries(fname):
    # load passed in hd5 file
    dd = soft_import("deepdish")
    return dd.io.load(fname)
