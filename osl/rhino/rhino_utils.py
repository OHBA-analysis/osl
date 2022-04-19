#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:24:11 2021

@author: woolrich
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import nibabel as nib
import os
import os.path as op
from mne import write_surface, Transform
from mne.transforms import read_trans
from mne.io.meas_info import _simplify_info
from mne.io.pick import pick_channels_cov, pick_info
from mne.source_estimate import _make_stc, _get_src_type
from mne.cov import Covariance, make_ad_hoc_cov
from mne.forward import _subject_from_forward
from mne.forward.forward import is_fixed_orient, _restrict_forward_to_src_sel
from mne.io.proj import make_projector, Projection
from mne.minimum_norm.inverse import _get_vertno, _prepare_forward
from mne.source_space import label_src_vertno_sel
from mne.utils import (verbose, _check_one_ch_type, _check_info_inv, check_fname,
                       _reg_pinv, _check_option, logger,
                       _pl, _check_src_normal, check_version, _sym_mat_pow, warn)

from mne.rank import compute_rank
from mne.minimum_norm.inverse import _check_depth
from mne.beamformer._compute_beamformer import (
    _prepare_beamformer_input, _compute_power, _reduce_leadfield_rank,
    _compute_beamformer, _check_src_type, Beamformer, _proj_whiten_data,
    _sym_inv_sm)

from scipy.ndimage import generic_filter
from scipy.spatial import KDTree
from scipy import LowLevelCallable

import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer


#############################################################################
def get_gridstep(fwd):
    """
    Get gridstep (i.e. spatial resolution of dipole grid) in mm from forward model

    Inputs
    ------
    fwd: mne.Forward
            Forward model

    Outputs
    -------
    gridstep: int
            Spatial resolution of dipole grid in mm

    """

    rr = fwd['src'][0]['rr']
    return _get_gridstep(rr)


#############################################################################
def _get_gridstep(coords):
    store = []
    for ii in range(coords.shape[0]):
        store.append(np.sqrt(np.sum(np.square(coords[ii, :] - coords[0, :]))))
    store = np.asarray(store)
    gridstep = int(np.round(np.min(store[np.where(store > 0)]) * 1000))

    return gridstep


#############################################################################
def niimask2indexpointcloud(nii_fname, volindex=None):
    """
    Takes in a nii.gz mask file name (which equals zero for background and neq zero for
    the mask) and returns the mask as a 3 x npoints point cloud

    Input
    -----
        nii_fname: string
                A nii.gz mask file name
                (with zero for background, and !=0 for the mask)
        volindex: int
                    Volume index, used if nii_mask is a 4D file

    Output
    ------
        pc: nd.array
                3 x npoints point cloud as voxel indices
    """

    vol = nib.load(nii_fname).get_fdata()

    if len(vol.shape) == 4 and volindex is not None:
        vol = vol[:, :, :, volindex]

    if not len(vol.shape) == 3:
        Exception(
            "nii_mask must be a 3D volume, or nii_mask must be a 4D volume with volindex specifying a volume index")

    # Turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
    pc = np.asarray(np.where(vol != 0))

    return pc


#############################################################################
def niimask2mmpointcloud(nii_mask, volindex=None):
    """
    Takes in a nii.gz mask (which equals zero for background and neq zero for
    the mask) and returns the mask as a 3 x npoints point cloud in
    native space in mm's

    Input:
        nii_mask: string
                    A nii.gz mask file name or the [x,y,z] volume
                    (with zero for background, and !=0 for the mask)
        volindex: int
                    Volume index, used if nii_mask is a 4D file

    Return:
        pc - 3 x npoints point cloud as mm in native space (using sform)
        values - npoints values

    """

    vol = nib.load(nii_mask).get_fdata()

    if len(vol.shape) == 4 and volindex is not None:
        vol = vol[:, :, :, volindex]

    if not len(vol.shape) == 3:
        Exception(
            "nii_mask must be a 3D volume, or nii_mask must be a 4D volume with volindex specifying a volume index")

    # Turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
    pc_nativeindex = np.asarray(np.where(vol != 0))

    values = np.asarray(vol[vol != 0])

    # Move from native voxel indices to native space coordinates (in mm)
    pc = xform_points(_get_sform(nii_mask)['trans'], pc_nativeindex)

    return pc, values


#############################################################################
def _closest_node(node, nodes):
    """
    Find nearest node in nodes to the passed in node.
    Returns the index to the nearest node in nodes.
    """

    if len(nodes) == 1:
        nodes = np.reshape(nodes, [-1, 1])

    kdtree = KDTree(nodes)
    distance, index = kdtree.query(node)

    return index, distance


#############################################################################

def _get_sform(nii_file):
    # sform allows mapping from simple voxel index cordinates
    # (e.g. from 0 to 256) in scanner space to continuous coordinates (in mm)

    # sformcode = os.popen('fslorient -getsformcode {}'.format(
    # nii_file)).read().strip()

    sformcode = int(nib.load(nii_file).header['sform_code'])

    if sformcode == 1 or sformcode == 4:
        sform = nib.load(nii_file).header.get_sform()
    else:
        raise ValueError('sform code for {} is {}, and needs to be 4 or 1'.format(nii_file, sformcode))

    sform = Transform('mri_voxel', 'mri', sform)
    return sform


#############################################################################

def _get_mni_sform(nii_file):
    # sform allows mapping from simple voxel index cordinates
    # (e.g. from 0 to 256) in scanner space to continuous coordinates (in mm)

    # sformcode = os.popen('fslorient -getsformcode {}'.format(
    # nii_file)).read().strip()

    sformcode = int(nib.load(nii_file).header['sform_code'])

    if sformcode == 1 or sformcode == 4:
        sform = nib.load(nii_file).header.get_sform()
    else:
        raise ValueError('sform code for {} is {}, and needs to be 4 or 1'.format(nii_file, sformcode))

    sform = Transform('unknown', 'mni_tal', sform)
    return sform


#############################################################################

def _get_orient(nii_file):
    cmd = 'fslorient -getorient {}'.format(nii_file)

    # use os.popen rather than os.system as we want to return a value, 
    # note that this will wait until the read() works before continuing. 
    # Without the read() the code will continue without waiting for the 
    # system call to finish
    orient = os.popen(cmd).read().strip()

    return orient


#############################################################################

# def _majority(buffer, required_majority):
#    return buffer.sum() >= required_majority

# See https://ilovesymposia.com/2017/03/12/scipys-new-lowlevelcallable-is-a-game-changer/
# Numba cfunc that takes in:
# a double pointer pointing to the values within the footprint,
# a pointer-sized integer that specifies the number of values in the footprint,
# a double pointer for the result, and
# a void pointer, which could point to additional parameters

@cfunc(intc(CPointer(float64), intp,
            CPointer(float64), voidptr))
def majority(values_ptr, len_values, result, data):
    values = carray(values_ptr, (len_values,), dtype=float64)
    required_majority = 14  # in 3D we have 27 voxels in total
    result[0] = values.sum() >= required_majority

    return 1


#############################################################################

def _binary_majority3d(img):
    # Set a pixel to 1 if a required majority (default=14) or more pixels 
    # in its 3x3x3 neighborhood are 1, otherwise, set the pixel to 0.
    # img is a 3D binary image

    if img.dtype != 'bool':
        raise ValueError('binary_majority3d(img) requires img to be binary')

    if len(img.shape) != 3:
        raise ValueError('binary_majority3d(img) requires img to be 3D')

    imgout = generic_filter(img,
                            LowLevelCallable(majority.ctypes),
                            size=3,
                            ).astype(int)

    return imgout


#############################################################################
def rigid_transform_3D(B, A):
    """
    Calculate affine transform from points in A to point in B
    Input:
        A: numpy.ndarray
                3 x num_points. Set of points to register from
        B: numpy.ndarray
                3 x num_points. Set of points to register to
    Returns:
        xform: numpy.ndarray
                Calculated affine transform

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

    t = -R @ centroid_A + centroid_B

    xform = np.eye(4)
    xform[0:3, 0:3] = R
    xform[0:3, -1] = np.reshape(t, (1, -1))

    return xform


#############################################################################
def xform_points(xform, pnts):
    """
    Applies homogenous linear transformation to an array of 3D coordinates

    Input:
        xform: numpy.ndarray
                4x4 matrix containing the affine transform
        pnts: numpy.ndarray
                points to transform, should be 3 x num_points

    Returns:
        newpnts: numpy.ndarray
                pnts following the xform, will be 3 x num_points

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


#############################################################################
def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions

    Inputs
    ------
      A: numpy.ndarray
            Nxm numpy array of corresponding points
      B: numpy.ndarray
            Nxm numpy array of corresponding points

    Outputs
    -------
      T: numpy.ndarray
            (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
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


#############################################################################
def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src

    Inputs
    ------
        src: numpy.ndarray
                Nxm array of points
        dst: numpy.ndarray
                Nxm array of points
    Outputs
    -------
        distances: numpy.ndarray
                Euclidean distances of the nearest neighbor
        indices: numpy.ndarray
                dst indices of the nearest neighbor
    """

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


#############################################################################
def icp(A, B, init_pose=None, max_iterations=50, tolerance=0.0001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Inputs
    ------
        A: numpy.ndarray
            Nxm numpy array of source mD points
        B: numpy.ndarray
            Nxm numpy array of destination mD point
        init_pose: numpy.ndarray
            (m+1)x(m+1) homogeneous transformation
        max_iterations: int
            Exit algorithm after max_iterations
        tolerance: float
            Convergence criteria
    Outputs
    -------
        T: numpy.ndarray
                (4 x 4) Final homogeneous transformation that maps A on to B
        distances: numpy.ndarray
                Euclidean distances (errors) of the nearest neighbor
        i: float
                Number of iterations to converge

    From: https://github.com/ClayFlannigan/icp/blob/master/icp.py
    """

    # assert A.shape == B.shape

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

    # import pdb; pdb.pdb.set_trace()

    kdtree = KDTree(dst[:m, :].T)

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        # distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        distances, indices = kdtree.query(src[:m, :].T)

        # import pdb; pdb.pdb.set_trace()

        # compute the transformation between the current source and nearest destination points
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


#############################################################################
def rhino_icp(smri_headshape_polhemus,
              polhemus_headshape_polhemus,
              Ninits=10):
    """
    Runs Iterative Closest Point with multiple initialisations

    Inputs
    ------
    smri_headshape_polhemus: numpy.ndarray
                    [3 x N] locations of the
                    Headshape points in polehumus space
                    (i.e. MRI scalp surface)

    polhemus_headshape_polhemus: numpy.ndarray
                    [3 x N] locations of the
                    Polhemus headshape points in polhemus space

    Ninits: int
            Number of random initialisations to perform

    Outputs
    -------

    xform: numpy.ndarray
            [4 x 4] rigid transformation matrix mapping data2 to data

    Based on Matlab version from Adam Baker 2014
    """

    # These are the "destination" points that are held static
    data1 = smri_headshape_polhemus

    # These are the "source" points that will be moved around
    data2 = polhemus_headshape_polhemus

    err_old = np.Infinity
    err = np.zeros(Ninits)

    Mr = np.eye(4)

    incremental = False
    if incremental:
        Mr_total = np.eye(4)

    data2r = data2

    for init in range(Ninits):

        Mi, distances, i = icp(data2r.T, data1.T)

        # RMS error
        e = np.sqrt(np.mean(np.square(distances)))
        err[init] = e

        if err[init] < err_old:

            print('ICP found better xform, error={}'.format(e))

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
                ax = plt.axes(projection='3d')
                ax.scatter(data1[0, 0:-1:10], data1[1, 0:-1:10], data1[2, 0:-1:10], c='blue', marker='.', s=1)
                ax.scatter(data2[0, :], data2[1, :], data2[2, :], c='red', marker='o', s=5)
                ax.scatter(data2r[0, :], data2r[1, :], data2r[2, :], c='green', marker='o', s=5)
                ax.scatter(data2r_icp[0, :], data2r_icp[1, :], data2r_icp[2, :], c='yellow', marker='o', s=5)
                plt.show()
                plt.draw()

        #####
        # Give the registration a kick...
        a = (np.random.uniform() - 0.5) * np.pi / 6
        b = (np.random.uniform() - 0.5) * np.pi / 6
        c = (np.random.uniform() - 0.5) * np.pi / 6

        Rx = np.array([
            (1, 0, 0),
            (0, np.cos(a), -np.sin(a)),
            (0, np.sin(a), np.cos(a))])
        Ry = np.array([
            (np.cos(b), 0, np.sin(b)),
            (0, 1, 0),
            (-np.sin(b), 0, np.cos(b))])
        Rz = np.array([
            (np.cos(c), -np.sin(c), 0),
            (np.sin(c), np.cos(c), 0),
            (0, 0, 1)])

        T = 10 * np.array((np.random.uniform() - 0.5, np.random.uniform() - 0.5, np.random.uniform() - 0.5))
        Mr = np.eye(4)
        Mr[0:3, 0:3] = Rx @ Ry @ Rz
        Mr[0:3, -1] = np.reshape(T, (1, -1))

        if incremental:
            data2r = Mr @ Mr_total @ np.vstack((data2, np.ones((1, data2.shape[1]))))
        else:
            data2r = Mr @ np.vstack((data2, np.ones((1, data2.shape[1]))))

        data2r = data2r[0:3, :]

        #####

    return xform, err, err_old


#############################################################################
def create_freesurfer_mesh(infile,
                           surf_outfile,
                           xform_mri_voxel2mri,
                           nii_mesh_file=None):
    """
    Creates surface mesh in .surf format and in native mri space in mm
    from infile

    Inputs
    ------
     infile: string
         Either:
            1) .nii.gz file containing zero's for background and one's for surface
            2) .vtk file generated by bet_surf (in which case the path to the
            strutural MRI, smri_file, must be included as an input)

     surf_outfile: string
            Path to the .surf file generated, containing the surface
            mesh in mm

     xform_mri_voxel2mri: numpy.ndarray
            4x4 array
            Transform from voxel indices to native/mri mm

     nii_mesh_file: string
            Path to the niftii mesh file that is the niftii equivalent
            of vtk file passed in as infile (only needed if infile
            is a vtk file)
    """

    pth, name = op.split(infile)
    name, ext = op.splitext(name)

    if ext == '.gz':

        print('Creating surface mesh for {} .....'.format(infile))

        name, ext = op.splitext(name)
        if ext != '.nii':
            raise ValueError('Invalid infile. Needs to be a .nii.gz or .vtk file')

        # convert to point cloud in voxel indices
        nii_nativeindex = niimask2indexpointcloud(infile)

        # print('Num of vertices to create mesh from = {}'.format(nii_nativeindex.shape[1]))

        step = 1
        nii_native = xform_points(xform_mri_voxel2mri, nii_nativeindex[:, 0:-1:step])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(nii_native.T)
        pcd.estimate_normals()
        # to obtain a consistent normal orientation
        pcd.orient_normals_towards_camera_location(pcd.get_center())

        # or you might want to flip the normals to make them point outward, not mandatory
        pcd.normals = o3d.utility.Vector3dVector(- np.asarray(pcd.normals))

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)[0]

        # mesh = mesh.simplify_quadric_decimation(nii_nativeindex.shape[1])

        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles).astype(int)

        # output in freesurfer file format
        write_surface(surf_outfile, verts, tris, file_format='freesurfer', overwrite=True)
    elif ext == '.vtk':

        if nii_mesh_file is None:
            raise ValueError('You must specify a nii_mesh_file (niftii format), if \
infile format is vtk')

        rrs_native, tris_native = _get_vtk_mesh_native(infile, nii_mesh_file)

        write_surface(surf_outfile, rrs_native, tris_native, file_format='freesurfer', overwrite=True)
    else:
        raise ValueError('Invalid infile. Needs to be a .nii.gz or .vtk file')

    # print('Written new surface mesh: {}'.format(surf_outfile))


#############################################################################
def _get_vtk_mesh_native(vtk_mesh_file, nii_mesh_file):
    """
    Returns mesh rrs in native space in mm and the meash tris for the passed
    in vtk_mesh_file

    nii_mesh_file needs to be the corresponding niftii file from bet
    that corresponds to the same mesh as in vtk_mesh_file
    """

    data = pd.read_csv(vtk_mesh_file, delim_whitespace=True)

    num_rrs = int(data.iloc[3, 1])

    # these will be in voxel index space
    rrs_flirtcoords = data.iloc[4:num_rrs + 4, 0:3].to_numpy().astype(np.float64)

    # move to from flirtcoords mm to mri mm (native) space
    xform_flirtcoords2nii = _get_flirtcoords2native_xform(nii_mesh_file)
    rrs_nii = xform_points(xform_flirtcoords2nii, rrs_flirtcoords.T).T

    num_tris = int(data.iloc[num_rrs + 4, 1])
    tris_nii = data.iloc[num_rrs + 5:num_rrs + 5 + num_tris, 1:4].to_numpy().astype(int)

    return rrs_nii, tris_nii


def _get_flirtcoords2native_xform(nii_mesh_file):
    # Returns xform_flirtcoords2native transform that transforms from
    # flirtcoords space in mm into native space in mm, where the passed in 
    # nii_mesh_file specifies the native space 
    #
    # Note that for some reason flirt outputs transforms of the form:
    # flirt_mni2mri = mri2flirtcoords x mni2mri x flirtcoords2mni
    #
    # and bet_surf outputs the .vtk file vertex values
    # in the same flirtcoords mm coordinate system. 
    #
    # See the bet_surf manual 
    # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#betsurf
    #
    # If the image has radiological ordering ( see fslorient ) then the mm 
    # co-ordinates are the voxel co-ordinates scaled by the mm voxel sizes.
    #
    # i.e. ( x_mm = x_dim * x ) 
    # where x_mm are the flirtcoords coords in mm, 
    # x is the voxel co-ordinate and x_dim is the voxel size in mm."
    #
    #####

    # We will assume orientation of the smri is RADIOLOGICAL as RHINO will have
    # made the smri the same orientation as the standard brain nii.
    # But let's just double check that is the case:
    smri_orient = _get_orient(nii_mesh_file)
    if smri_orient != 'RADIOLOGICAL':
        raise ValueError('Orientation of file must be RADIOLOGICAL,\
please check output of: fslorient -getorient {}'.format(nii_mesh_file))

    xform_nativevox2native = _get_sform(nii_mesh_file)['trans']
    dims = np.append(nib.load(nii_mesh_file).header.get_zooms(), 1)

    # Then calc xform based on x_mm = x_dim * x (see above)
    xform_flirtcoords2nativevox = np.diag(1.0 / dims)
    xform_flirtcoords2native = xform_nativevox2native @ xform_flirtcoords2nativevox

    return xform_flirtcoords2native


#############################################################################
def _transform_vtk_mesh(vtk_mesh_file_in, nii_mesh_file_in, out_vtk_file, nii_mesh_file_out, xform_file):
    # Outputs mesh to out_vtk_file, which is the result of applying the
    # transform xform to vtk_mesh_file_in
    # 
    # nii_mesh_file_in needs to be the corresponding niftii file from bet
    # that corresponds to the same mesh as in vtk_mesh_file_in
    #
    # nii_mesh_file_out needs to be the corresponding niftii file from bet
    # that corresponds to the same mesh as in out_vtk_file

    rrs_in, tris_in = _get_vtk_mesh_native(vtk_mesh_file_in, nii_mesh_file_in)

    xform_flirtcoords2native_out = _get_flirtcoords2native_xform(nii_mesh_file_out)

    xform = read_trans(xform_file)['trans']

    overall_xform = np.linalg.inv(xform_flirtcoords2native_out) @ xform

    # rrs_in are in native nii_in space in mm
    # transform them using the passed in xform
    rrs_out = xform_points(overall_xform, rrs_in.T).T

    data = pd.read_csv(vtk_mesh_file_in, delim_whitespace=True)

    num_rrs = int(data.iloc[3, 1])
    data.iloc[4:num_rrs + 4, 0:3] = rrs_out

    # write new vtk file
    data.to_csv(out_vtk_file, sep=' ', index=False)


#############################################################################

def _get_xform_from_flirt_xform(flirt_xform, nii_mesh_file_in, nii_mesh_file_out):
    # Get mm coordinates to mm coordinates xform using a known flirt xform
    #
    # Note that we need to do this as flirt xforms include an extra xform
    # based on the voxel dimensions (see _get_flirtcoords2native_xform )

    flirtcoords2native_xform_in = _get_flirtcoords2native_xform(nii_mesh_file_in)
    flirtcoords2native_xform_out = _get_flirtcoords2native_xform(nii_mesh_file_out)

    xform = flirtcoords2native_xform_out @ flirt_xform @ np.linalg.inv(flirtcoords2native_xform_in)

    return xform


#############################################################################

def _get_flirt_xform_between_axes(from_nii, target_nii):
    # Computes flirt xform that moves from_nii to have voxel indices on the
    # same axis as  the voxel indices for target_nii.
    #
    # Note that this is NOT the same as registration, i.e. the images are not
    # aligned. In fact the actual coordinates (in mm) are unchanged.
    # It is instead about putting from_nii onto the same axes
    # so that the voxel INDICES are comparable. This is achieved by using a
    # transform that sets the sform of from_nii to be the same as target_nii
    # without changing the actual coordinates (in mm).
    # Transform needed to do this is:
    #   from2targetaxes = inv(targetvox2target) * fromvox2from
    #
    # In more detail:
    # We need the sform for the transformed from_nii 
    # to be the same as the sform for the target_nii, without changing the 
    # actual coordinates (in mm). 
    # In other words, we need:
    # fromvox2from * from_nii_vox = targetvox2target * from_nii_target_vox
    # where
    #   fromvox2from is sform for from_nii (i.e. converts from voxel indices to
    #       voxel coords in mm)
    #   and targetvox2target is sform for target_nii
    #   and from_nii_vox are the voxel indices for from_nii
    #   and from_nii_target_vox are the voxel indices for from_nii when 
    #       transformed onto the target axis.
    #
    # => from_nii_target_vox = from2targetaxes * from_nii_vox
    # where 
    #   from2targetaxes = inv(targetvox2target) * fromvox2from

    to2tovox = np.linalg.inv(_get_sform(target_nii)['trans'])
    fromvox2from = _get_sform(from_nii)['trans']

    from2to = to2tovox @ fromvox2from

    return from2to


#############################################################################

def plot_polhemus_points(txt_fnames, colors=None, scales=None,
                         markers=None, alphas=None):
    plt.figure()
    ax = plt.axes(projection='3d')

    for ss in range(len(txt_fnames)):

        if alphas is None:
            alpha = 1
        else:
            alpha = alphas[ss]

        if colors is None:
            color = (0.5, 0.5, 0.5)
        else:
            color = colors[ss]

        if scales is None:
            scale = 10
        else:
            scale = scales[ss]

        if markers is None:
            marker = 1
        else:
            marker = markers[ss]

        pnts = np.loadtxt(txt_fnames[ss])

        ax.scatter(pnts[0, ], pnts[1, ], pnts[2, ],
                   color=color, s=scale, alpha=alpha, marker=marker)

#############################################################################

@ verbose
def _make_lcmv(info, forward, data_cov,
                reg=0.05, noise_cov=None, label=None,
                pick_ori=None, rank='info',
                noise_rank='info',
                weight_norm='unit-noise-gain-invariant',
                reduce_rank=False, depth=None, inversion='matrix', verbose=None):
    """Compute LCMV spatial filter.

    Parameters
    ----------
    info : instance of Info
        The measurement info to specify the channels to include.
        Bad channels in info['bads'] are not used.
    forward : instance of Forward
        Forward operator.
    data_cov : instance of Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    noise_cov : instance of Covariance
        The noise covariance. If provided, whitening will be done. Providing a
        noise covariance is mandatory if you mix sensor types, e.g.
        gradiometers with magnetometers or EEG with MEG.
    label : instance of Label
        Restricts the LCMV solution to a given label.

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
            'noise_cov' : instance of Covariance | None
                The noise covariance matrix used to compute the beamformer.
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
                direction of maximum power at each source location.
            'inversion' : 'single' | 'matrix'
                Whether the spatial filters were computed for each dipole
                separately or jointly for all dipoles at each vertex using a
                matrix inversion.

    Notes
    -----
    Rhino version of mne.beamformer.make_lcmv

    Note that code that is different to mne.beamformer.make_lcmv is labelled
    with MWW

    The original reference is :footcite:`VanVeenEtAl1997`.

    To obtain the Sekihara unit-noise-gain vector beamformer, you should use
    ``weight_norm='unit-noise-gain', pick_ori='vector'`` followed by
    :meth:`vec_stc.project('pca', src) <mne.VectorSourceEstimate.project>`.

    .. versionchanged:: 0.21
       The computations were extensively reworked, and the default for
       ``weight_norm`` was set to ``'unit-noise-gain-invariant'``.

    References
    ----------
    .. footbibliography::
    """
    # check number of sensor types present in the data and ensure a noise cov
    info = _simplify_info(info)
    noise_cov, _, allow_mismatch = _check_one_ch_type(
        'lcmv', info, forward, data_cov, noise_cov)
    # XXX we need this extra picking step (can't just rely on minimum norm's
    # because there can be a mismatch. Should probably add an extra arg to
    # _prepare_beamformer_input at some point (later)
    picks = _check_info_inv(info, forward, data_cov, noise_cov)
    info = pick_info(info, picks)

    data_rank = compute_rank(data_cov, rank=rank, info=info)
    noise_rank = compute_rank(noise_cov, rank=noise_rank, info=info)

    if False:  # MWW
        for key in data_rank:
            if (key not in noise_rank or data_rank[key] != noise_rank[key]) and \
                    not allow_mismatch:
                raise ValueError('%s data rank (%s) did not match the noise '
                                 'rank (%s)'
                                 % (key, data_rank[key],
                                    noise_rank.get(key, None)))
    # MWW
    # del noise_rank
    rank = data_rank
    logger.info('Making LCMV beamformer with data cov rank %s' % (rank,))
    # MWW added:
    logger.info('Making LCMV beamformer with noise cov rank %s' % (noise_rank,))

    del data_rank
    depth = _check_depth(depth, 'depth_sparse')
    if inversion == 'single':
        depth['combine_xyz'] = False

    # MWW
    is_free_ori, info, proj, vertno, G, whitener, nn, orient_std = \
        _prepare_beamformer_input(
            info, forward, label, pick_ori, noise_cov=noise_cov, rank=noise_rank,
            pca=False, **depth)

    ch_names = list(info['ch_names'])

    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov._get_square()
    if 'estimator' in data_cov:
        del data_cov['estimator']
    rank_int = sum(rank.values())
    del rank

    # compute spatial filter
    n_orient = 3 if is_free_ori else 1
    W, max_power_ori = _compute_beamformer(
        G, Cm, reg, n_orient, weight_norm, pick_ori, reduce_rank, rank_int,
        inversion=inversion, nn=nn, orient_std=orient_std,
        whitener=whitener)

    # get src type to store with filters for _make_stc
    src_type = _get_src_type(forward['src'], vertno)

    # get subject to store with filters
    subject_from = _subject_from_forward(forward)

    # Is the computed beamformer a scalar or vector beamformer?
    is_free_ori = is_free_ori if pick_ori in [None, 'vector'] else False
    is_ssp = bool(info['projs'])

    filters = Beamformer(
        kind='LCMV', weights=W, data_cov=data_cov, noise_cov=noise_cov,
        whitener=whitener, weight_norm=weight_norm, pick_ori=pick_ori,
        ch_names=ch_names, proj=proj, is_ssp=is_ssp, vertices=vertno,
        is_free_ori=is_free_ori, n_sources=forward['nsource'],
        src_type=src_type, source_nn=forward['source_nn'].copy(),
        subject=subject_from, rank=rank_int, max_power_ori=max_power_ori,
        inversion=inversion)

    return filters

#############################################################################

def _compute_beamformer(G, Cm, reg, n_orient, weight_norm, pick_ori,
                        reduce_rank, rank, inversion, nn, orient_std,
                        whitener):
    """Compute a spatial beamformer filter (LCMV or DICS).

    For more detailed information on the parameters, see the docstrings of
    `make_lcmv` and `make_dics`.

    Parameters
    ----------
    G : ndarray, shape (n_dipoles, n_channels)
        The leadfield.
    Cm : ndarray, shape (n_channels, n_channels)
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
    nn : ndarray, shape (n_dipoles, 3)
        The source normals.
    orient_std : ndarray, shape (n_dipoles,)
        The std of the orientation prior used in weighting the lead fields.
    whitener : ndarray, shape (n_channels, n_channels)
        The whitener.

    Returns
    -------
    W : ndarray, shape (n_dipoles, n_channels)
        The beamformer filter weights.
    """
    _check_option('weight_norm', weight_norm,
                  ['unit-noise-gain-invariant', 'unit-noise-gain',
                   'nai', None])

    # Whiten the data covariance
    Cm = whitener @ Cm @ whitener.T.conj()
    # Restore to properly Hermitian as large whitening coefs can have bad
    # rounding error

    Cm[:] = (Cm + Cm.T.conj()) / 2.

    assert Cm.shape == (G.shape[0],) * 2
    s, _ = np.linalg.eigh(Cm)
    if not (s >= -s.max() * 1e-7).all():
        # This shouldn't ever happen, but just in case
        warn('data covariance does not appear to be positive semidefinite, '
             'results will likely be incorrect')
    # Tikhonov regularization using reg parameter to control for
    # trade-off between spatial resolution and noise sensitivity
    # eq. 25 in Gross and Ioannides, 1999 Phys. Med. Biol. 44 2081
    Cm_inv, loading_factor, rank = _reg_pinv(Cm, reg, rank)

    assert orient_std.shape == (G.shape[1],)
    n_sources = G.shape[1] // n_orient
    assert nn.shape == (n_sources, 3)

    logger.info('Computing beamformer filters for %d source%s'
                % (n_sources, _pl(n_sources)))
    n_channels = G.shape[0]
    assert n_orient in (3, 1)
    Gk = np.reshape(G.T, (n_sources, n_orient, n_channels)).transpose(0, 2, 1)
    assert Gk.shape == (n_sources, n_channels, n_orient)
    sk = np.reshape(orient_std, (n_sources, n_orient))
    del G, orient_std
    pinv_kwargs = dict()
    if check_version('numpy', '1.17'):
        pinv_kwargs['hermitian'] = True

    _check_option('reduce_rank', reduce_rank, (True, False))

    # inversion of the denominator
    _check_option('inversion', inversion, ('matrix', 'single'))
    if inversion == 'single' and n_orient > 1 and pick_ori == 'vector' and \
            weight_norm == 'unit-noise-gain-invariant':
        raise ValueError(
            'Cannot use pick_ori="vector" with inversion="single" and '
            'weight_norm="unit-noise-gain-invariant"')
    if reduce_rank and inversion == 'single':
        raise ValueError('reduce_rank cannot be used with inversion="single"; '
                         'consider using inversion="matrix" if you have a '
                         'rank-deficient forward model (i.e., from a sphere '
                         'model with MEG channels), otherwise consider using '
                         'reduce_rank=False')
    if n_orient > 1:
        _, Gk_s, _ = np.linalg.svd(Gk, full_matrices=False)
        assert Gk_s.shape == (n_sources, n_orient)
        if not reduce_rank and (Gk_s[:, 0] > 1e6 * Gk_s[:, 2]).any():
            raise ValueError(
                'Singular matrix detected when estimating spatial filters. '
                'Consider reducing the rank of the forward operator by using '
                'reduce_rank=True.')
        del Gk_s

    #
    # 1. Reduce rank of the lead field
    #
    if reduce_rank:
        Gk = _reduce_leadfield_rank(Gk)

    def _compute_bf_terms(Gk, Cm_inv):
        bf_numer = np.matmul(Gk.swapaxes(-2, -1).conj(), Cm_inv)
        bf_denom = np.matmul(bf_numer, Gk)
        return bf_numer, bf_denom

    #
    # 2. Reorient lead field in direction of max power or normal
    #
    if pick_ori == 'max-power' or pick_ori == 'max-power-pre-weight-norm':
        assert n_orient == 3
        _, bf_denom = _compute_bf_terms(Gk, Cm_inv)

        if pick_ori == 'max-power':
            if weight_norm is None:
                ori_numer = np.eye(n_orient)[np.newaxis]
                ori_denom = bf_denom
            else:
                # compute power, cf Sekihara & Nagarajan 2008, eq. 4.47
                ori_numer = bf_denom
                # Cm_inv should be Hermitian so no need for .T.conj()
                ori_denom = np.matmul(
                    np.matmul(Gk.swapaxes(-2, -1).conj(), Cm_inv @ Cm_inv), Gk)

            ori_denom_inv = _sym_inv_sm(ori_denom, reduce_rank, inversion, sk)
            ori_pick = np.matmul(ori_denom_inv, ori_numer)

        # MWW
        else:  # pick_ori == 'max-power-pre-weight-norm':

            # Compute power, see eq 5 in Brookes et al, Optimising experimental design for MEG beamformer imaging, Neuroimage 2008
            # This optimises the orientation by maximising the power
            # BEFORE any weight normalisation is performed
            ori_pick = _sym_inv_sm(bf_denom, reduce_rank, inversion, sk)

        assert ori_pick.shape == (n_sources, n_orient, n_orient)

        # pick eigenvector that corresponds to maximum eigenvalue:
        eig_vals, eig_vecs = np.linalg.eig(ori_pick.real)  # not Hermitian!
        # sort eigenvectors by eigenvalues for picking:
        order = np.argsort(np.abs(eig_vals), axis=-1)
        # eig_vals = np.take_along_axis(eig_vals, order, axis=-1)
        max_power_ori = eig_vecs[np.arange(len(eig_vecs)), :, order[:, -1]]
        assert max_power_ori.shape == (n_sources, n_orient)

        # set the (otherwise arbitrary) sign to match the normal
        signs = np.sign(np.sum(max_power_ori * nn, axis=1, keepdims=True))
        signs[signs == 0] = 1.
        max_power_ori *= signs

        # Compute the lead field for the optimal orientation,
        # and adjust numer/denom

        Gk = np.matmul(Gk, max_power_ori[..., np.newaxis])

        n_orient = 1
    else:
        max_power_ori = None
        if pick_ori == 'normal':
            Gk = Gk[..., 2:3]
            n_orient = 1

    #
    # 3. Compute numerator and denominator of beamformer formula (unit-gain)
    #

    bf_numer, bf_denom = _compute_bf_terms(Gk, Cm_inv)
    assert bf_denom.shape == (n_sources,) + (n_orient,) * 2
    assert bf_numer.shape == (n_sources, n_orient, n_channels)
    del Gk  # lead field has been adjusted and should not be used anymore

    #
    # 4. Invert the denominator
    #

    # Here W is W_ug, i.e.:
    # G.T @ Cm_inv / (G.T @ Cm_inv @ G)
    bf_denom_inv = _sym_inv_sm(bf_denom, reduce_rank, inversion, sk)
    assert bf_denom_inv.shape == (n_sources, n_orient, n_orient)
    W = np.matmul(bf_denom_inv, bf_numer)
    assert W.shape == (n_sources, n_orient, n_channels)
    del bf_denom_inv, sk

    #
    # 5. Re-scale filter weights according to the selected weight_norm
    #

    # Weight normalization is done by computing, for each source::
    #
    #     W_ung = W_ug / sqrt(W_ug @ W_ug.T)
    #
    # with W_ung referring to the unit-noise-gain (weight normalized) filter
    # and W_ug referring to the above-calculated unit-gain filter stored in W.

    if weight_norm is not None:
        # Three different ways to calculate the normalization factors here.
        # Only matters when in vector mode, as otherwise n_orient == 1 and
        # they are all equivalent. Sekihara 2008 says to use
        #
        # In MNE < 0.21, we just used the Frobenius matrix norm:
        #
        #    noise_norm = np.linalg.norm(W, axis=(1, 2), keepdims=True)
        #    assert noise_norm.shape == (n_sources, 1, 1)
        #    W /= noise_norm
        #
        # Sekihara 2008 says to use sqrt(diag(W_ug @ W_ug.T)), which is not
        # rotation invariant:
        if weight_norm in ('unit-noise-gain', 'nai'):
            noise_norm = np.matmul(W, W.swapaxes(-2, -1).conj()).real
            noise_norm = np.reshape(  # np.diag operation over last two axes
                noise_norm, (n_sources, -1, 1))[:, ::n_orient + 1]
            np.sqrt(noise_norm, out=noise_norm)
            noise_norm[noise_norm == 0] = np.inf
            assert noise_norm.shape == (n_sources, n_orient, 1)
            W /= noise_norm
        else:
            assert weight_norm == 'unit-noise-gain-invariant'
            # Here we use sqrtm. The shortcut:
            #
            #    use = W
            #
            # ... does not match the direct route (it is rotated!), so we'll
            # use the direct one to match FieldTrip:
            use = bf_numer
            inner = np.matmul(use, use.swapaxes(-2, -1).conj())
            W = np.matmul(_sym_mat_pow(inner, -0.5), use)
            noise_norm = 1.

        if weight_norm == 'nai':
            # Estimate noise level based on covariance matrix, taking the
            # first eigenvalue that falls outside the signal subspace or the
            # loading factor used during regularization, whichever is largest.
            if rank > len(Cm):
                # Covariance matrix is full rank, no noise subspace!
                # Use the loading factor as noise ceiling.
                if loading_factor == 0:
                    raise RuntimeError(
                        'Cannot compute noise subspace with a full-rank '
                        'covariance matrix and no regularization. Try '
                        'manually specifying the rank of the covariance '
                        'matrix or using regularization.')
                noise = loading_factor
            else:
                noise, _ = np.linalg.eigh(Cm)
                noise = noise[-rank]
                noise = max(noise, loading_factor)
            W /= np.sqrt(noise)

    W = W.reshape(n_sources * n_orient, n_channels)
    logger.info('Filter computation complete')
    return W, max_power_ori

