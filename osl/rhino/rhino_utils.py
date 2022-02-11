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

from scipy.ndimage import generic_filter
from scipy.spatial import KDTree
from scipy import LowLevelCallable

import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

#############################################################################
def niimask2indexpointcloud(nii_fname):
    '''
    Takes in a nii.gz mask file name (with neq one for background and ones for
    the mask) and returns the mask as a 3 x npoints point cloud
    
    Input:
        nii_fname - a nii.gz mask file name 
                    (with zero for background, and !=0 for the mask)
        
    Return:
        pc - 3 x npoints point cloud as voxel indices
    '''
    
    vol = nib.load(nii_fname).get_fdata()
    
    # Turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
    pc = np.asarray(np.where(vol!=0))
        
    return pc

#############################################################################
def niimask2mmpointcloud(mask_nii_fname):
    '''
    Takes in a nii.gz mask file name (with neq one for background and ones for
    the mask) and returns the mask as a 3 x npoints point cloud in
    native space in mm's
    
    Input:
        mask_nii_fname - a nii.gz mask file name 
                    (with zero for background, and !=0 for the mask)
        
    Return:
        pc - 3 x npoints point cloud as mm in native space (using sform)
        values - npoints values
        
    '''
    
    vol = nib.load(mask_nii_fname).get_fdata()
    
    # Turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
    pc_nativeindex = np.asarray(np.where(vol!=0))
    
    values = np.asarray(vol[vol!=0])
    
    # Move from native voxel indices to native space coordinates (in mm)
    pc = xform_points(_get_sform(mask_nii_fname)['trans'], pc_nativeindex)

    return pc, values

#############################################################################
def _closest_node(node, nodes):
        
    '''
    Find nearest node in nodes to the passed in node.
    Returns the index to the nearest node in nodes.
    '''
    
    if len(nodes)==1:
        nodes=np.reshape(nodes,[-1,1])
        
    kdtree = KDTree(nodes)
    distance, index = kdtree.query(node)
    
    return index, distance

#############################################################################

def _get_sform(nii_file):
    
    # sform allows mapping from simple voxel index cordinates 
    # (e.g. from 0 to 256) in scanner space to continuous coordinates (in mm)
    
    #sformcode = os.popen('fslorient -getsformcode {}'.format(
    #nii_file)).read().strip()
    
    sformcode = int(nib.load(nii_file).header['sform_code'])
    
    if sformcode == 1 or sformcode == 4:        
        sform = nib.load(nii_file).header.get_sform()
    else:
        raise ValueError('sform code for {} is {}, and needs to be 4 or 1'.format(nii_file, sformcode) )
        
    sform = Transform('mri_voxel', 'mri', sform)
    return sform

#############################################################################

def _get_mni_sform(nii_file):
    
    # sform allows mapping from simple voxel index cordinates 
    # (e.g. from 0 to 256) in scanner space to continuous coordinates (in mm)
    
    #sformcode = os.popen('fslorient -getsformcode {}'.format(
    #nii_file)).read().strip()
    
    sformcode = int(nib.load(nii_file).header['sform_code'])
    
    if sformcode == 1 or sformcode == 4:
        sform = nib.load(nii_file).header.get_sform()
    else:
        raise ValueError('sform code for {} is {}, and needs to be 4 or 1'.format(nii_file, sformcode) )
        
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

#def _majority(buffer, required_majority):
#    return buffer.sum() >= required_majority

from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer
 
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
     required_majority = 14 # in 3D we have 27 voxels in total
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

    '''
    Calculate affine transform from points in A to point in B
    Input: 
        A and B expected to be 3 x num_points
        B      - set of points to register to
        A      - set of points to register from
    Returns:
        xform
    see http://nghiaho.com/?page_id=671
    '''
    
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
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    xform = np.eye(4)
    xform[0:3,0:3]=R
    xform[0:3,-1]=np.reshape(t, (1,-1))

    return xform

#############################################################################
def xform_points(xform, pnts):
    '''
    Applies homogenous linear transformation to an array of 3D coordinates
    
    Input:
        xform - 4x4 matri containing the affine transform
        pnts - points to transform should be 3 x num_points
    
    Returns:
        newpnts - pnts following the xform, will be 3 x num_points
    
    '''
    if len(pnts.shape)==1:
        pnts=np.reshape(pnts,[-1,1])
        
    num_rows, num_cols = pnts.shape
    if num_rows != 3:
        raise Exception(f"pnts is not 3xN, it is {num_rows}x{num_cols}")
       
    pnts = np.concatenate((pnts,np.ones([1,pnts.shape[1]])),axis=0)
        
    newpnts = xform @ pnts
    newpnts = newpnts[0:3,:]
    
    return newpnts

#############################################################################
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    '''

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
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    
    return T

#############################################################################
def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    #assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

#############################################################################
def icp(A, B, init_pose=None, max_iterations=50, tolerance=0.0001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    
    From: https://github.com/ClayFlannigan/icp/blob/master/icp.py
    '''

    #assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    
    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    #import pdb; pdb.pdb.set_trace()

    kdtree = KDTree(dst[:m,:].T)

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        #distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        distances, indices = kdtree.query(src[:m,:].T)
        
        #import pdb; pdb.pdb.set_trace()        
             
        # compute the transformation between the current source and nearest destination points
        T = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check RMS error
        mean_error = np.sqrt(np.mean(np.square(distances)))
        
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


#############################################################################
def rhino_icp(smri_headshape_polhemus, 
              polhemus_headshape_polhemus, 
              Ninits=10):

    '''
    Runs Iterative Closest Point with multiple initialisations
    
    REQUIRED INPUTS:
    
    smri_headshape_polhemus - [3 x N] locations of the 
                                Headshape points in polehumus space
                                (i.e. MRI scalp surface)
                
    polhemus_headshape_polhemus - [3 x N] locations of the 
                                    Polhemus headshape points in polhemus space
                                    
    
    OPTIONAL INPUTS:
    
    Ninits     - Number of random initialisations to perform (default 10)
    
    OUTPUTS:
    
    xform      - [4 x 4] rigid transformation matrix mapping data2 to data
    
    Based on Matlab version from Adam Baker 2014
    '''
           
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
        
    data2r  = data2    
    
    for init in range (Ninits):
    
        Mi, distances, i = icp(data2r.T, data1.T)
        
        # RMS error
        e=np.sqrt(np.mean(np.square(distances)))
        err[init] = e
             
        if err[init] < err_old:

            print('ICP found better xform, error={}'.format(e))        
            
            err_old = e
            
            if incremental:
                Mr_total= Mr @ Mr_total
                xform   = Mi @ Mr_total
            else:
                xform   = Mi @ Mr
                            
            if False:
                import matplotlib.pyplot as plt
        
                # after ICP
                data2r_icp = xform_points(xform,data2)
                plt.figure(frameon=False)
                ax = plt.axes(projection='3d')
                ax.scatter(data1[0,0:-1:10],data1[1,0:-1:10], data1[2,0:-1:10],c='blue',marker ='.',s=1)              
                ax.scatter(data2[0,:],data2[1,:], data2[2,:],c='red',marker ='o',s=5)
                ax.scatter(data2r[0,:],data2r[1,:],data2r[2,:],c='green',marker ='o',s=5)
                ax.scatter(data2r_icp[0,:],data2r_icp[1,:],data2r_icp[2,:],c='yellow',marker ='o',s=5)
                plt.show()        
                plt.draw()
                        
        
        #####
        # Give the registration a kick...
        a = (np.random.uniform()-0.5)*np.pi/3
        b = (np.random.uniform()-0.5)*np.pi/3
        c = (np.random.uniform()-0.5)*np.pi/3

        Rx = np.array([
              (1, 0, 0),
              (0, np.cos(a), -np.sin(a)),
              (0, np.sin(a), np.cos(a)) ])
        Ry = np.array([
              (np.cos(b), 0, np.sin(b)),
              (0, 1, 0),
              (-np.sin(b), 0, np.cos(b)) ])
        Rz = np.array([
              (np.cos(c), -np.sin(c), 0),
              (np.sin(c), np.cos(c), 0),
              (0, 0, 1) ])
                
        T  = 15*np.array((np.random.uniform()-0.5, np.random.uniform()-0.5, np.random.uniform()-0.5))
        Mr = np.eye(4)
        Mr[0:3,0:3]= Rx@Ry@Rz
        Mr[0:3,-1]=np.reshape(T, (1,-1))      
        
        if incremental:        
            data2r = Mr @ Mr_total @ np.vstack((data2, np.ones((1,data2.shape[1]))))
        else:
            data2r = Mr @ np.vstack((data2, np.ones((1,data2.shape[1]))))
        
        data2r = data2r[0:3,:]
                
        #####
        
    return xform, err, err_old

#############################################################################
def create_freesurfer_mesh(infile, 
                surf_outfile,
                xform_mri_voxel2mri,
                nii_mesh_file=None,
                overwrite=True):
    '''
    Creates surface mesh in .surf format and in native mri space in mm 
    from infile
     
    Inputs
    ------
     infile -  string
         Either:
            1) .nii.gz file containing zero's for background and one's for surface
            2) .vtk file generated by bet_surf (in which case the path to the
            strutural MRI, smri_file, must be included as an input)
    
     surf_outfile - string
             Path to the .surf file generated, containing the surface
             mesh in mm
    
     xform_mri_voxel2mri -  4x4 numpy array
             Transform from voxel indices to native/mri mm
                
     nii_mesh_file - string
             Path to the niftii mesh file that is the niftii equivalent
             of vtk file passed in as infile (only needed if infile 
             is a vtk file)
    '''
        
    overwrite = True
    if os.path.isfile(surf_outfile) is False or overwrite is True:
    
        pth, name = op.split(infile)
        name, ext = op.splitext(name) 
                
        if ext == '.gz':
            
            print('Creating surface mesh for {} .....'.format(infile))
            
            name, ext = op.splitext(name) 
            if ext != '.nii':
                raise ValueError('Invalid infile. Needs to be a .nii.gz or .vtk file')
            
            # convert to point cloud in voxel indices
            nii_nativeindex = niimask2indexpointcloud(infile)
                        
            #print('Num of vertices to create mesh from = {}'.format(nii_nativeindex.shape[1]))
            
            if nii_nativeindex.shape[1]>100000:
                step = 10
                radius_multiple = 3
            else:
                step = 1
                radius_multiple = 1.5
                           
            nii_native = xform_points(xform_mri_voxel2mri, nii_nativeindex[:,0:-1:step])
                
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(nii_native.T)
            pcd.estimate_normals()
            
            # estimate radius for rolling ball
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = radius_multiple * avg_dist   
            
            #import pdb; pdb.pdb.set_trace()

            if True:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=2, linear_fit=False)[0]
            
                bbox = pcd.get_axis_aligned_bounding_box()
                mesh = mesh.crop(bbox)
            else:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                           pcd,
                           o3d.utility.DoubleVector([radius*0.75, radius, radius * 2]))
                
            mesh = mesh.simplify_quadric_decimation(100000)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            verts = np.asarray(mesh.vertices)
            tris = np.asarray(mesh.triangles).astype(int)            
            
            # output in freesurfer file format
            write_surface(surf_outfile, verts, tris, file_format='freesurfer', overwrite=overwrite)
            
            if False:
                from mayavi import mlab
                
                # Visualize the points
                pts = mlab.points3d(nii_native[0,:], nii_native[1,:], nii_native[2,:], scale_mode='none', scale_factor=0.2)
                
                # Create and visualize the mesh
                mesh = mlab.pipeline.delaunay2d(pts)
                surf = mlab.pipeline.surface(mesh)
            
        elif ext == '.vtk':
            
            if nii_mesh_file==None:
                raise ValueError('You must specify a nii_mesh_file (niftii format), if \
infile format is vtk')
                        
            rrs_native, tris_native  = _get_vtk_mesh_native(infile, nii_mesh_file)
            
            write_surface(surf_outfile, rrs_native, tris_native, file_format='freesurfer', overwrite=overwrite)
        else:
            raise ValueError('Invalid infile. Needs to be a .nii.gz or .vtk file')
       
        #print('Written new surface mesh: {}'.format(surf_outfile))

#############################################################################
def _get_vtk_mesh_native(vtk_mesh_file, nii_mesh_file):
    
    # Returns mesh rrs in native space in mm and the meash tris for the passed
    # in vtk_mesh_file
    #
    # nii_mesh_file needs to be the corresponding niftii file from bet
    # that corresponds to the same mesh as in vtk_mesh_file
       
    data = pd.read_csv(vtk_mesh_file, delim_whitespace=True)
    
    num_rrs = int(data.iloc[3,1])
    
    # these will be in voxel index space
    rrs_flirtcoords = data.iloc[4:num_rrs+4,0:3].to_numpy().astype(np.float64)
                
    # move to from flirtcoords mm to mri mm (native) space
    xform_flirtcoords2nii = _get_flirtcoords2native_xform(nii_mesh_file)    
    rrs_nii = xform_points(xform_flirtcoords2nii, rrs_flirtcoords.T).T
    
    num_tris = int(data.iloc[num_rrs+4,1])
    tris_nii = data.iloc[num_rrs+5:num_rrs+5+num_tris,1:4].to_numpy().astype(int)
    
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
    dims = np.append(nib.load(nii_mesh_file).header.get_zooms(),1)
        
    # Then calc xform based on x_mm = x_dim * x (see above)
    xform_flirtcoords2nativevox = np.diag(1.0/dims)
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

    num_rrs = int(data.iloc[3,1])
    data.iloc[4:num_rrs+4,0:3] = rrs_out
    
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

def plot_polhemus_points(txt_fnames, colors = None, scales = None, 
                                        markers = None, alphas = None):
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
        
        ax.scatter(pnts[0, ],
               pnts[1, ],
               pnts[2, ],
               color=color, s=scale, alpha=alpha, marker=marker)
            
            