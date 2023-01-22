"""RHINO utilities.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import os.path as op
import subprocess
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
import nibabel as nib
import pandas as pd
from scipy import LowLevelCallable
from scipy.ndimage import generic_filter
from scipy.spatial import KDTree

from mne import Transform
from mne.transforms import read_trans
from mne.surface import write_surface

from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer

import logging
logging.getLogger("numba").setLevel(logging.WARNING)

from osl.source_recon.beamforming import transform_recon_timeseries
from osl.utils.logger import log_or_print
from osl.utils import soft_import


def system_call(cmd, verbose=False):
    if verbose:
        log_or_print(cmd)
    subprocess.call(cmd, shell=True)


def get_gridstep(coords):
    """Get gridstep (i.e. spatial resolution of dipole grid) in mm.

    Parameters
    ----------
    coords : numpy.ndarray
        Coordinates.

    Returns
    -------
    gridstep: int
        Spatial resolution of dipole grid in mm.
    """
    store = []
    for ii in range(coords.shape[0]):
        store.append(np.sqrt(np.sum(np.square(coords[ii, :] - coords[0, :]))))
    store = np.asarray(store)
    gridstep = int(np.round(np.min(store[np.where(store > 0)]) * 1000))
    return gridstep


def niimask2indexpointcloud(nii_fname, volindex=None):
    """Takes in a nii.gz mask file name (which equals zero for background and
    != zero for the mask) and returns the mask as a 3 x npoints point cloud.

    Parameters
    ----------
    nii_fname : string
        A nii.gz mask file name (with zero for background, and !=0 for the mask).
    volindex : int
        Volume index, used if nii_mask is a 4D file.

    Returns
    -------
    pc : numpy.ndarray
        3 x npoints point cloud as voxel indices.
    """

    vol = nib.load(nii_fname).get_fdata()

    if len(vol.shape) == 4 and volindex is not None:
        vol = vol[:, :, :, volindex]

    if not len(vol.shape) == 3:
        Exception(
            "nii_mask must be a 3D volume, or nii_mask must be a 4D volume with volindex specifying a volume index"
        )

    # Turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
    pc = np.asarray(np.where(vol != 0))

    return pc


def niimask2mmpointcloud(nii_mask, volindex=None):
    """Takes in a nii.gz mask (which equals zero for background and neq zero for
    the mask) and returns the mask as a 3 x npoints point cloud in native space
    in mm's.

    Parameters
    ----------
    nii_mask : string
        A nii.gz mask file name or the [x,y,z] volume
        (with zero for background, and !=0 for the mask).
    volindex : int
        Volume index, used if nii_mask is a 4D file.

    Returns
    -------
    pc : numpy.ndarray
        3 x npoints point cloud as mm in native space (using sform).
    values : numpy.ndarray
         npoints values.
    """

    vol = nib.load(nii_mask).get_fdata()

    if len(vol.shape) == 4 and volindex is not None:
        vol = vol[:, :, :, volindex]

    if not len(vol.shape) == 3:
        Exception(
            "nii_mask must be a 3D volume, or nii_mask must be a 4D volume with volindex specifying a volume index"
        )

    # Turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
    pc_nativeindex = np.asarray(np.where(vol != 0))

    values = np.asarray(vol[vol != 0])

    # Move from native voxel indices to native space coordinates (in mm)
    pc = xform_points(_get_sform(nii_mask)["trans"], pc_nativeindex)

    return pc, values


def _closest_node(node, nodes):
    """Find nearest node in nodes to the passed in node.

    Returns
    -------
    index : int
        Index to the nearest node in nodes.
    distance : float
        Distance.
    """

    if len(nodes) == 1:
        nodes = np.reshape(nodes, [-1, 1])

    kdtree = KDTree(nodes)
    distance, index = kdtree.query(node)

    return index, distance


def _get_vol_info_from_nii(mri):
    """Read volume info from an MRI file.

    Parameters
    ----------
    mri : str
        Path to MRI file.

    Returns
    -------
    out : dict
        Dictionary with keys 'mri_width', 'mri_height', 'mri_depth'
        and 'mri_volume_name'.
    """
    dims = nib.load(mri).get_fdata().shape
    out = dict(
        mri_width=dims[0],
        mri_height=dims[1],
        mri_depth=dims[2],
        mri_volume_name=mri,
    )
    return out


def _get_sform(nii_file):
    """
    sform allows mapping from simple voxel index cordinates
    (e.g. from 0 to 256) in scanner space to continuous coordinates (in mm)

    sformcode = os.popen('fslorient -getsformcode {}'.format(
    nii_file)).read().strip()
    """

    sformcode = int(nib.load(nii_file).header["sform_code"])

    if sformcode == 1 or sformcode == 4:
        sform = nib.load(nii_file).header.get_sform()
    else:
        raise ValueError(
            "sform code for {} is {}, and needs to be 4 or 1".format(
                nii_file, sformcode
            )
        )

    sform = Transform("mri_voxel", "mri", sform)
    return sform


def _get_mni_sform(nii_file):
    """
    sform allows mapping from simple voxel index cordinates
    (e.g. from 0 to 256) in scanner space to continuous coordinates (in mm)

    sformcode = os.popen('fslorient -getsformcode {}'.format(
    nii_file)).read().strip()
    """

    sformcode = int(nib.load(nii_file).header["sform_code"])

    if sformcode == 1 or sformcode == 4:
        sform = nib.load(nii_file).header.get_sform()
    else:
        raise ValueError(
            "sform code for {} is {}, and needs to be 4 or 1".format(
                nii_file, sformcode
            )
        )

    sform = Transform("unknown", "mni_tal", sform)
    return sform


def _get_orient(nii_file):
    cmd = "fslorient -getorient {}".format(nii_file)

    # use os.popen rather than os.system as we want to return a value,
    # note that this will wait until the read() works before continuing.
    # Without the read() the code will continue without waiting for the
    # system call to finish
    orient = os.popen(cmd).read().strip()

    return orient


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def majority(values_ptr, len_values, result, data):
    """
    def _majority(buffer, required_majority):
       return buffer.sum() >= required_majority

    See https://ilovesymposia.com/2017/03/12/scipys-new-lowlevelcallable-is-a-game-changer/
    Numba cfunc that takes in:
    a double pointer pointing to the values within the footprint,
    a pointer-sized integer that specifies the number of values in the footprint,
    a double pointer for the result, and
    a void pointer, which could point to additional parameters
    """
    values = carray(values_ptr, (len_values,), dtype=float64)
    required_majority = 14  # in 3D we have 27 voxels in total
    result[0] = values.sum() >= required_majority
    return 1


def _binary_majority3d(img):
    """
    Set a pixel to 1 if a required majority (default=14) or more pixels
    in its 3x3x3 neighborhood are 1, otherwise, set the pixel to 0.
    img is a 3D binary image
    """

    if img.dtype != "bool":
        raise ValueError("binary_majority3d(img) requires img to be binary")

    if len(img.shape) != 3:
        raise ValueError("binary_majority3d(img) requires img to be 3D")

    imgout = generic_filter(
        img,
        LowLevelCallable(majority.ctypes),
        size=3,
    ).astype(int)

    return imgout


def rigid_transform_3D(B, A, compute_scaling=False):
    """Calculate affine transform from points in A to point in B.

    Parameters
    ----------
    A : numpy.ndarray
        3 x num_points. Set of points to register from.
    B : numpy.ndarray
        3 x num_points. Set of points to register to.

    compute_scaling : bool
        Do we compute a scaling on top of rotation and translation?

    Returns
    -------
    xform : numpy.ndarray
        Calculated affine transform, does not include scaling.
    scaling_xform : numpy.ndarray
        Calculated scaling transform (a diagonal 4x4 matrix),
        does not include rotation or translation.

    see http://nghiaho.com/?page_id=671
    """

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    scaling_xform = np.eye(4)

    if compute_scaling:
        # Bm = C @ R @ Am + error
        # where C = c I is a scalar scaling
        RAm = R @ Am
        U2, S2, V2t = np.linalg.svd(Bm @ np.linalg.pinv(RAm))

        # take average of eigenvalues, accounting for the rank
        S2 = np.identity(3) * np.mean(S2[S2 > 1e-9])

        scaling_xform[0:3, 0:3] = S2

    t = -R @ centroid_A + centroid_B

    xform = np.eye(4)
    xform[0:3, 0:3] = R
    xform[0:3, -1] = np.reshape(t, (1, -1))

    return xform, scaling_xform


def xform_points(xform, pnts):
    """Applies homogenous linear transformation to an array of 3D coordinates.

    Parameters
    ----------
    xform : numpy.ndarray
        4x4 matrix containing the affine transform.
    pnts : numpy.ndarray
        points to transform, should be 3 x num_points.

    Returns
    -------
    newpnts : numpy.ndarray
        pnts following the xform, will be 3 x num_points.
    """
    if len(pnts.shape) == 1:
        pnts = np.reshape(pnts, [-1, 1])

    num_rows, num_cols = pnts.shape
    if num_rows != 3:
        raise Exception(f"pnts is not 3xN, it is {num_rows}x{num_cols}")

    pnts = np.concatenate((pnts, np.ones([1, pnts.shape[1]])), axis=0)

    newpnts = xform @ pnts
    newpnts = newpnts[0:3, :]

    return newpnts


def best_fit_transform(A, B):
    """Calculates the least-squares best-fit transform that maps corresponding
    points A to B in m spatial dimensions.

    Parameters
    ----------
    A : numpy.ndarray
        Nxm numpy array of corresponding points.
    B : numpy.ndarray
        Nxm numpy array of corresponding points.

    Outputs
    -------
    T : numpy.ndarray
        (m+1)x(m+1) homogeneous transformation matrix that maps A on to B.
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T


def nearest_neighbor(src, dst):
    """Find the nearest (Euclidean) neighbor in dst for each point in src.

    Parameters
    ----------
    src : numpy.ndarray
        Nxm array of points.
    dst : numpy.ndarray
        Nxm array of points.

    Returns
    -------
    distances : numpy.ndarray
        Euclidean distances of the nearest neighbor.
    indices : numpy.ndarray
        dst indices of the nearest neighbor.
    """

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=50, tolerance=0.0001):
    """The Iterative Closest Point method: finds best-fit transform that maps
    points A on to points B.

    Parameters
    ----------
    A : numpy.ndarray
        Nxm numpy array of source mD points.
    B : numpy.ndarray
        Nxm numpy array of destination mD point.
    init_pose : numpy.ndarray
        (m+1)x(m+1) homogeneous transformation.
    max_iterations : int
        Exit algorithm after max_iterations.
    tolerance : float
        Convergence criteria.

    Returns
    -------
    T : numpy.ndarray
        (4 x 4) Final homogeneous transformation that maps A on to B.
    distances : numpy.ndarray
        Euclidean distances (errors) of the nearest neighbor.
    i : float
        Number of iterations to converge.

    Notes
    -----
    From: https://github.com/ClayFlannigan/icp/blob/master/icp.py
    """

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    kdtree = KDTree(dst[:m, :].T)

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        # distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        distances, indices = kdtree.query(src[:m, :].T)

        # compute the transformation between the current source and nearest
        # destination points
        T = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check RMS error
        mean_error = np.sqrt(np.mean(np.square(distances)))

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


def rhino_icp(smri_headshape_polhemus, polhemus_headshape_polhemus, n_init=10):
    """Runs Iterative Closest Point (ICP) with multiple initialisations.

    Parameters
    ----------
    smri_headshape_polhemus : numpy.ndarray
        [3 x N] locations of the Headshape points in polehumus space
        (i.e. MRI scalp surface).
    polhemus_headshape_polhemus : numpy.ndarray
        [3 x N] locations of the Polhemus headshape points in polhemus space.
    n_init : int
        Number of random initialisations to perform.

    Returns
    -------
    xform : numpy.ndarray
        [4 x 4] rigid transformation matrix mapping data2 to data.

    Notes
    -----
    Based on Matlab version from Adam Baker 2014.
    """

    # These are the "destination" points that are held static
    data1 = smri_headshape_polhemus

    # These are the "source" points that will be moved around
    data2 = polhemus_headshape_polhemus

    err_old = np.Infinity
    err = np.zeros(n_init)

    Mr = np.eye(4)

    incremental = False
    if incremental:
        Mr_total = np.eye(4)

    data2r = data2

    for init in range(n_init):

        Mi, distances, i = icp(data2r.T, data1.T)

        # RMS error
        e = np.sqrt(np.mean(np.square(distances)))
        err[init] = e

        if err[init] < err_old:

            log_or_print("ICP found better xform, error={}".format(e))

            err_old = e

            if incremental:
                Mr_total = Mr @ Mr_total
                xform = Mi @ Mr_total
            else:
                xform = Mi @ Mr

            if False:
                import matplotlib.pyplot as plt

                # after ICP
                data2r_icp = xform_points(xform, data2)
                plt.figure(frameon=False)
                ax = plt.axes(projection="3d")
                ax.scatter(
                    data1[0, 0:-1:10],
                    data1[1, 0:-1:10],
                    data1[2, 0:-1:10],
                    c="blue",
                    marker=".",
                    s=1,
                )
                ax.scatter(
                    data2[0, :], data2[1, :], data2[2, :], c="red", marker="o", s=5
                )
                ax.scatter(
                    data2r[0, :], data2r[1, :], data2r[2, :], c="green", marker="o", s=5
                )
                ax.scatter(
                    data2r_icp[0, :],
                    data2r_icp[1, :],
                    data2r_icp[2, :],
                    c="yellow",
                    marker="o",
                    s=5,
                )
                plt.show()
                plt.draw()

        # Give the registration a kick...
        a = (np.random.uniform() - 0.5) * np.pi / 6
        b = (np.random.uniform() - 0.5) * np.pi / 6
        c = (np.random.uniform() - 0.5) * np.pi / 6

        Rx = np.array(
            [(1, 0, 0), (0, np.cos(a), -np.sin(a)), (0, np.sin(a), np.cos(a))]
        )
        Ry = np.array(
            [(np.cos(b), 0, np.sin(b)), (0, 1, 0), (-np.sin(b), 0, np.cos(b))]
        )
        Rz = np.array(
            [(np.cos(c), -np.sin(c), 0), (np.sin(c), np.cos(c), 0), (0, 0, 1)]
        )

        T = 10 * np.array(
            (
                np.random.uniform() - 0.5,
                np.random.uniform() - 0.5,
                np.random.uniform() - 0.5,
            )
        )
        Mr = np.eye(4)
        Mr[0:3, 0:3] = Rx @ Ry @ Rz
        Mr[0:3, -1] = np.reshape(T, (1, -1))

        if incremental:
            data2r = Mr @ Mr_total @ np.vstack((data2, np.ones((1, data2.shape[1]))))
        else:
            data2r = Mr @ np.vstack((data2, np.ones((1, data2.shape[1]))))

        data2r = data2r[0:3, :]

    return xform, err, err_old


def _get_vtk_mesh_native(vtk_mesh_file, nii_mesh_file):
    """
    Returns mesh rrs in native space in mm and the mesh tris for the passed
    in vtk_mesh_file

    nii_mesh_file needs to be the corresponding niftii file from bet
    that corresponds to the same mesh as in vtk_mesh_file
    """

    data = pd.read_csv(vtk_mesh_file, delim_whitespace=True)

    num_rrs = int(data.iloc[3, 1])

    # these will be in voxel index space
    rrs_flirtcoords = data.iloc[4 : num_rrs + 4, 0:3].to_numpy().astype(np.float64)

    # move from flirtcoords mm to mri mm (native) space
    xform_flirtcoords2nii = _get_flirtcoords2native_xform(nii_mesh_file)
    rrs_nii = xform_points(xform_flirtcoords2nii, rrs_flirtcoords.T).T

    num_tris = int(data.iloc[num_rrs + 4, 1])
    tris_nii = (
        data.iloc[num_rrs + 5 : num_rrs + 5 + num_tris, 1:4].to_numpy().astype(int)
    )

    return rrs_nii, tris_nii


def _get_flirtcoords2native_xform(nii_mesh_file):
    """
    Returns xform_flirtcoords2native transform that transforms from
    flirtcoords space in mm into native space in mm, where the passed in
    nii_mesh_file specifies the native space

    Note that for some reason flirt outputs transforms of the form:
    flirt_mni2mri = mri2flirtcoords x mni2mri x flirtcoords2mni

    and bet_surf outputs the .vtk file vertex values
    in the same flirtcoords mm coordinate system.

    See the bet_surf manual
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#betsurf

    If the image has radiological ordering ( see fslorient ) then the mm
    co-ordinates are the voxel co-ordinates scaled by the mm voxel sizes.

    i.e. ( x_mm = x_dim * x )
    where x_mm are the flirtcoords coords in mm,
    x is the voxel co-ordinate and x_dim is the voxel size in mm."
    """

    # We will assume orientation of the smri is RADIOLOGICAL as RHINO will have
    # made the smri the same orientation as the standard brain nii.
    # But let's just double check that is the case:
    smri_orient = _get_orient(nii_mesh_file)
    if smri_orient != "RADIOLOGICAL":
        raise ValueError(
            "Orientation of file must be RADIOLOGICAL,\
please check output of: fslorient -getorient {}".format(
                nii_mesh_file
            )
        )

    xform_nativevox2native = _get_sform(nii_mesh_file)["trans"]
    dims = np.append(nib.load(nii_mesh_file).header.get_zooms(), 1)

    # Then calc xform based on x_mm = x_dim * x (see above)
    xform_flirtcoords2nativevox = np.diag(1.0 / dims)
    xform_flirtcoords2native = xform_nativevox2native @ xform_flirtcoords2nativevox

    return xform_flirtcoords2native


def _transform_vtk_mesh(
    vtk_mesh_file_in, nii_mesh_file_in, out_vtk_file, nii_mesh_file_out, xform_file
):
    """
    Outputs mesh to out_vtk_file, which is the result of applying the
    transform xform to vtk_mesh_file_in

    nii_mesh_file_in needs to be the corresponding niftii file from bet
    that corresponds to the same mesh as in vtk_mesh_file_in

    nii_mesh_file_out needs to be the corresponding niftii file from bet
    that corresponds to the same mesh as in out_vtk_file
    """

    rrs_in, tris_in = _get_vtk_mesh_native(vtk_mesh_file_in, nii_mesh_file_in)

    xform_flirtcoords2native_out = _get_flirtcoords2native_xform(nii_mesh_file_out)

    if isinstance(xform_file, str):
        xform = read_trans(xform_file)["trans"]
    else:
        xform = xform_file

    overall_xform = np.linalg.inv(xform_flirtcoords2native_out) @ xform

    # rrs_in are in native nii_in space in mm
    # transform them using the passed in xform
    rrs_out = xform_points(overall_xform, rrs_in.T).T

    data = pd.read_csv(vtk_mesh_file_in, delim_whitespace=True)

    num_rrs = int(data.iloc[3, 1])
    data.iloc[4 : num_rrs + 4, 0:3] = rrs_out

    # write new vtk file
    data.to_csv(out_vtk_file, sep=" ", index=False)


def _get_mne_xform_from_flirt_xform(flirt_xform, nii_mesh_file_in, nii_mesh_file_out):
    """
    Returns a mm coordinates to mm coordinates MNE xform that corresponds to the
    passed in flirt xform.

    Note that we need to do this as flirt xforms include an extra xform
    based on the voxel dimensions (see _get_flirtcoords2native_xform).
    """

    flirtcoords2native_xform_in = _get_flirtcoords2native_xform(nii_mesh_file_in)
    flirtcoords2native_xform_out = _get_flirtcoords2native_xform(nii_mesh_file_out)

    xform = (
        flirtcoords2native_xform_out
        @ flirt_xform
        @ np.linalg.inv(flirtcoords2native_xform_in)
    )

    return xform


def _get_flirt_xform_between_axes(from_nii, target_nii):
    """
    Computes flirt xform that moves from_nii to have voxel indices on the
    same axis as  the voxel indices for target_nii.

    Note that this is NOT the same as registration, i.e. the images are not
    aligned. In fact the actual coordinates (in mm) are unchanged.
    It is instead about putting from_nii onto the same axes
    so that the voxel INDICES are comparable. This is achieved by using a
    transform that sets the sform of from_nii to be the same as target_nii
    without changing the actual coordinates (in mm).
    Transform needed to do this is:
      from2targetaxes = inv(targetvox2target) * fromvox2from

    In more detail:
    We need the sform for the transformed from_nii to be the same as the sform
    for the target_nii, without changing the actual coordinates (in mm).
    In other words, we need:
    fromvox2from * from_nii_vox = targetvox2target * from_nii_target_vox
    where
      fromvox2from is sform for from_nii (i.e. converts from voxel indices to
          voxel coords in mm)
      and targetvox2target is sform for target_nii
      and from_nii_vox are the voxel indices for from_nii
      and from_nii_target_vox are the voxel indices for from_nii when
          transformed onto the target axis.

    => from_nii_target_vox = from2targetaxes * from_nii_vox
    where
      from2targetaxes = inv(targetvox2target) * fromvox2from
    """

    to2tovox = np.linalg.inv(_get_sform(target_nii)["trans"])
    fromvox2from = _get_sform(from_nii)["trans"]

    from2to = to2tovox @ fromvox2from

    return from2to


def _timeseries2nii(
    timeseries, timeseries_coords, reference_mask_fname, out_nii_fname, times=None
):
    """Maps the (ndipoles,tpts) array of timeseries to the grid defined by
    reference_mask_fname and outputs them as a niftii file.

    Assumes the timeseries' dipoles correspond to those in reference_mask_fname.
    Both timeseries and reference_mask_fname are often output from
    rhino.transform_recon_timeseries.

    Parameters
    ----------
    timeseries : (ndipoles, ntpts) numpy.ndarray
        Time courses.
        Assumes the timeseries' dipoles correspond to those in reference_mask_fname.
        Typically derives from rhino.transform_recon_timeseries
    timeseries_coords : (ndipoles, 3) numpy.ndarray
        Coords in mm for dipoles corresponding to passed in timeseries
    reference_mask_fname : string
        A nii.gz mask file name
        (with zero for background, and !=0 for the mask)
        Assumes the mask was used to set dipoles for timeseries,
        typically derived from rhino.transform_recon_timeseries
    out_nii_fname : string
        output name of niftii file
    times : (ntpts, ) numpy.ndarray
        Times points in seconds.
        Assume that times are regularly spaced.
        Used to set nii file up correctly.

    Returns
    -------
    out_nii_fname : string
        Name of output niftii file
    """

    if len(timeseries.shape) == 1:
        timeseries = np.reshape(timeseries, [-1, 1])

    mni_nii_nib = nib.load(reference_mask_fname)
    coords_ind = niimask2indexpointcloud(reference_mask_fname).T
    coords_mni, tmp = niimask2mmpointcloud(reference_mask_fname)

    mni_nii_values = mni_nii_nib.get_fdata()
    mni_nii_values = np.zeros(np.append(mni_nii_values.shape, timeseries.shape[1]))

    kdtree = KDTree(coords_mni.T)
    gridstep = int(get_gridstep(coords_mni.T) / 1000)

    for ind in range(timeseries_coords.shape[1]):
        distance, index = kdtree.query(timeseries_coords[:, ind])
        # Exclude any timeseries_coords that are further than gridstep away
        # from the best matching coords_mni
        if distance < gridstep:
            mni_nii_values[
                coords_ind[ind, 0], coords_ind[ind, 1], coords_ind[ind, 2], :
            ] = timeseries[ind, :]

    # SAVE AS NIFTI
    vol_nii = nib.Nifti1Image(mni_nii_values, mni_nii_nib.affine)

    vol_nii.header.set_xyzt_units(2)  # mm
    if times is not None:
        vol_nii.header["pixdim"][4] = times[1] - times[0]
        vol_nii.header["toffset"] = -0.5
        vol_nii.header.set_xyzt_units(2, 8)  # mm and secs

    nib.save(vol_nii, out_nii_fname)

    return out_nii_fname


def recon_timeseries2niftii(
    subjects_dir,
    subject,
    recon_timeseries,
    out_nii_fname,
    spatial_resolution=None,
    reference_brain="mni",
    times=None,
):
    """Converts a (ndipoles,tpts) array of reconstructed timeseries (in
    head/polhemus space) to the corresponding dipoles in a standard brain
    grid in MNI space and outputs them as a niftii file.

    Parameters
    ----------
    subjects_dir : string
        Directory to find RHINO subject directories in.
    subject : string
        Subject name directory to find RHINO files in.
    recon_timeseries : (ndipoles, ntpts) np.ndarray
        Reconstructed time courses (in head (polhemus) space).
        Assumes that the dipoles are the same (and in the same order)
        as those in the forward model, coreg_filenames['forward_model_file'].
        Typically derive from the VolSourceEstimate's output by
        MNE source recon methods, e.g. mne.beamformer.apply_lcmv, obtained
        using a forward model generated by RHINO.
    spatial_resolution : int
        Resolution to use for the reference brain in mm
        (must be an integer, or will be cast to nearest int)
        If None, then the gridstep used in coreg_filenames['forward_model_file']
        is used.
    reference_brain : string, 'mni' or 'mri'
        'mni' indicates that the reference_brain is the stdbrain in MNI space.
        'mri' indicates that the reference_brain is the sMRI in native/mri space.
    times : (ntpts, ) np.ndarray
        Times points in seconds. Will assume that these are regularly spaced.

    Returns
    -------
    out_nii_fname : string
        Name of output niftii file.
    reference_brain_fname : string
        Niftii file name of standard brain mask in MNI space at requested
        resolution, int(stdbrain_resolution) (with zero for background, and
        !=0 for the mask).
    """

    if len(recon_timeseries.shape) == 1:
        recon_timeseries = np.reshape(recon_timeseries, [recon_timeseries.shape[0], 1])

    # ---------------------------------------------------
    # convert the recon_timeseries to the standard
    # space brain dipole grid at the specified resolution
    (
        recon_ts_out,
        reference_brain_fname,
        recon_coords_out,
        recon_indices,
    ) = transform_recon_timeseries(
        subjects_dir,
        subject,
        recon_timeseries=recon_timeseries,
        spatial_resolution=spatial_resolution,
        reference_brain=reference_brain,
    )

    # ----------------------------------
    # output recon_ts_out as niftii file
    out_nii_fname = _timeseries2nii(
        recon_ts_out,
        recon_coords_out,
        reference_brain_fname,
        out_nii_fname,
        times=times,
    )

    return out_nii_fname, reference_brain_fname


def save_or_show_renderer(renderer, filename):
    """Save or show a renderer.

    Parameters
    ----------
    renderer : mne.viz.backends._notebook._Renderer Object
        MNE renderer object.
    filename : str
        Filename to save display to (as an interactive html).
        Must have extension .html. If None we display the renderer.
    """
    if filename is None:
        renderer.show()
    else:
        allowed_extensions = [".html", ".pdf", ".svg", ".eps", ".ps", ".tex"]
        ext = Path(filename).suffix
        if ext not in allowed_extensions:
            raise ValueError(
                f"{ext} not allowed, please use one of the following: "
                + " ".join(allowed_extensions)
            )

        log_or_print(f"saving {filename}")
        if ext == ".html":
            renderer.figure.plotter.export_html(filename)
        elif ext in allowed_extensions:
            renderer.figure.plotter.save_graphic(filename)


def _create_freesurfer_meshes_from_bet_surfaces(filenames, xform_mri_voxel2mri):

    # Create sMRI-derived freesurfer surfaces in native/mri space in mm,
    # for use by forward modelling

    _create_freesurfer_mesh_from_bet_surface(
        infile=filenames["bet_inskull_mesh_vtk_file"],
        surf_outfile=filenames["bet_inskull_surf_file"],
        nii_mesh_file=filenames["bet_inskull_mesh_file"],
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )

    _create_freesurfer_mesh_from_bet_surface(
        infile=filenames["bet_outskull_mesh_vtk_file"],
        surf_outfile=filenames["bet_outskull_surf_file"],
        nii_mesh_file=filenames["bet_outskull_mesh_file"],
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )

    _create_freesurfer_mesh_from_bet_surface(
        infile=filenames["bet_outskin_mesh_vtk_file"],
        surf_outfile=filenames["bet_outskin_surf_file"],
        nii_mesh_file=filenames["bet_outskin_mesh_file"],
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )


def _create_freesurfer_mesh_from_bet_surface(
    infile, surf_outfile, xform_mri_voxel2mri, nii_mesh_file=None
):
    """Creates surface mesh in .surf format and in native mri space in mm
    from infile.

    Parameters
    ----------
    infile : string
    Either:
        1) .nii.gz file containing zero's for background and one's for surface
        2) .vtk file generated by bet_surf (in which case the path to the
        structural MRI, smri_file, must be included as an input)
    surf_outfile : string
        Path to the .surf file generated, containing the surface
        mesh in mm
    xform_mri_voxel2mri : numpy.ndarray
        4x4 array
        Transform from voxel indices to native/mri mm
    nii_mesh_file : string
        Path to the niftii mesh file that is the niftii equivalent
        of vtk file passed in as infile (only needed if infile
        is a vtk file)
    """

    pth, name = op.split(infile)
    name, ext = op.splitext(name)

    if ext == ".gz":
        print("Creating surface mesh for {} .....".format(infile))

        # Soft import raising an informative warning if not installed
        o3d = soft_import("open3d")

        name, ext = op.splitext(name)
        if ext != ".nii":
            raise ValueError("Invalid infile. Needs to be a .nii.gz or .vtk file")

        # convert to point cloud in voxel indices
        nii_nativeindex = niimask2indexpointcloud(infile)

        step = 1
        nii_native = xform_points(xform_mri_voxel2mri, nii_nativeindex[:, 0:-1:step])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(nii_native.T)
        pcd.estimate_normals()
        # to obtain a consistent normal orientation
        pcd.orient_normals_towards_camera_location(pcd.get_center())

        # or you might want to flip the normals to make them point outward, not mandatory
        pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)[
            0
        ]

        # mesh = mesh.simplify_quadric_decimation(nii_nativeindex.shape[1])

        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles).astype(int)

        # output in freesurfer file format
        write_surface(
            surf_outfile, verts, tris, file_format="freesurfer", overwrite=True
        )

    elif ext == ".vtk":
        if nii_mesh_file is None:
            raise ValueError(
                "You must specify a nii_mesh_file (niftii format), "
                + "if infile format is vtk"
            )

        rrs_native, tris_native = _get_vtk_mesh_native(infile, nii_mesh_file)

        write_surface(
            surf_outfile,
            rrs_native,
            tris_native,
            file_format="freesurfer",
            overwrite=True,
        )

    else:
        raise ValueError("Invalid infile. Needs to be a .nii.gz or .vtk file")


def _transform_bet_surfaces(flirt_xform_file, mne_xform_file, filenames, smri_file):

    # Transform betsurf mask/mesh using passed in flirt transform and mne transform
    for mesh_name in {"outskin_mesh", "inskull_mesh", "outskull_mesh"}:
        # xform mask
        system_call(
            "flirt -interp nearestneighbour -in {} -ref {} -applyxfm -init {} -out {}".format(
                op.join(filenames["basedir"], "flirt_" + mesh_name + ".nii.gz"),
                smri_file,
                flirt_xform_file,
                op.join(filenames["basedir"], mesh_name),
            )
        )

        # xform vtk mesh
        _transform_vtk_mesh(
            op.join(filenames["basedir"], "flirt_" + mesh_name + ".vtk"),
            op.join(filenames["basedir"], "flirt_" + mesh_name + ".nii.gz"),
            op.join(filenames["basedir"], mesh_name + ".vtk"),
            op.join(filenames["basedir"], mesh_name + ".nii.gz"),
            mne_xform_file,
        )
