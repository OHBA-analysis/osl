#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:35:48 2021

@author: woolrich
"""

import warnings
import os
import os.path as op
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from shutil import copyfile
from osl.rhino import rhino_utils
from scipy.ndimage import morphology
from sklearn.mixture import GaussianMixture
from cv2 import getStructuringElement, morphologyEx, MORPH_GRADIENT, MORPH_RECT

from mne.viz.backends.renderer import _get_renderer
from mne.transforms import write_trans, read_trans, apply_trans, _get_trans, \
    combine_transforms, Transform, rotation, invert_transform
from mne.viz._3d import _sensor_shape
from mne.forward import _create_meg_coils
from mne.io.pick import pick_types
from mne.io import _loc_to_coil_trans, read_info, read_raw
from mne import read_epochs, read_forward_solution, make_bem_model, \
    make_bem_solution, make_forward_solution, write_forward_solution
from mne.io.constants import FIFF
from mne.bem import ConductorModel, read_bem_solution


#############################################################################

def get_surfaces_filenames(subjects_dir, subject):
    '''
    Generates a a dict of files generated and used by rhino.compute_surfaces. 

    Inputs
    ------
        subjects_dir - string
                Directory to put RHINO subject dirs in.
                Files will be in subjects_dir/subject/rhino/surfaces/                        
        subject - string
                Subject name dir to put RHINO files in.
                Files will be in subjects_dir/subject/rhino/surfaces/

    Returns
    -------
        filenames - dict
                A dict of files generated and used by rhino.compute_surfaces. 
                
                Note that  due to the unusal naming conventions used by BET:
                 - bet_inskull_*_file is actually the brain surface 
                 - bet_outskull_*_file is actually the inner skull surface
                 - bet_outskin_*_file is the outer skin/scalp surface  
    '''

    filenames = []

    # create rhino filenames
    basefilename = op.join(subjects_dir, subject, 'rhino')

    if not os.path.isdir(basefilename):
        os.mkdir(basefilename)

    basefilename = op.join(basefilename, 'surfaces')

    if not os.path.isdir(basefilename):
        os.mkdir(basefilename)

    filenames = {
        'basefilename': basefilename,
        'smri_file': op.join(basefilename, 'smri.nii.gz'),
        'mni2mri_flirt_file': op.join(basefilename, 'mni2mri_flirt.txt'),
        'mni_mri_t_file': op.join(basefilename, 'mni_mri-trans.fif'),
        'bet_smri_basefile': op.join(basefilename, 'bet_smri'),
        'bet_outskin_mesh_file': op.join(basefilename, 'outskin_mesh.nii.gz'),
        'bet_outskin_mesh_vtk_file': op.join(basefilename, 'outskin_mesh.vtk'),
        'bet_outskin_surf_file': op.join(basefilename, 'outskin_surf.surf'),
        'bet_outskin_plus_nose_mesh_file': op.join(basefilename, 'outskin_plus_nose_mesh.nii.gz'),
        'bet_outskin_plus_nose_surf_file': op.join(basefilename, 'outskin_plus_nose_surf.surf'),
        'bet_inskull_mesh_file': op.join(basefilename, 'inskull_mesh.nii.gz'),
        'bet_inskull_mesh_vtk_file': op.join(basefilename, 'inskull_mesh.vtk'),
        'bet_inskull_surf_file': op.join(basefilename, 'inskull_surf.surf'),
        'bet_outskull_mesh_file': op.join(basefilename, 'outskull_mesh.nii.gz'),
        'bet_outskull_mesh_vtk_file': op.join(basefilename, 'outskull_mesh.vtk'),
        'bet_outskull_surf_file': op.join(basefilename, 'outskull_surf.surf'),
        'std_brain': os.environ['FSLDIR'] + '/data/standard/MNI152_T1_1mm_brain.nii.gz',
        'std_brain_bigfov': os.environ['FSLDIR'] + '/data/standard/MNI152_T1_1mm_BigFoV_facemask.nii.gz'
    }

    return filenames


#############################################################################

def get_coreg_filenames(subjects_dir, subject):
    '''
    Generates a a dict of files generated and used by RHINO. 

    Inputs
    ------
        subjects_dir - string
                Directory to put RHINO subject dirs in.
                Files will be in subjects_dir/subject/rhino/coreg/                        
        subject - string
                Subject name dir to put RHINO files in.
                Files will be in subjects_dir/subject/rhino/coreg/

    Returns
    -------
        filenames - dict
                A dict of files generated and used by RHINO. 
    '''

    filenames = []

    # create rhino filenames
    basefilename = op.join(subjects_dir, subject, 'rhino')

    if not os.path.isdir(basefilename):
        os.mkdir(basefilename)

    basefilename = op.join(basefilename, 'coreg')
    if not os.path.isdir(basefilename):
        os.mkdir(basefilename)

    filenames = {
        'basefilename': basefilename,
        'fif_file': op.join(basefilename, 'data-raw.fif'),
        'smri_file': op.join(basefilename, 'smri.nii.gz'),
        'head_mri_t_file': op.join(basefilename, 'head_mri-trans.fif'),
        'ctf_head_mri_t_file': op.join(basefilename, 'ctf_head_mri-trans.fif'),
        'mrivoxel_mri_t_file': op.join(basefilename, 'mrivoxel_mri_t_file-trans.fif'),
        'smri_nasion_file': op.join(basefilename, 'smri_nasion.txt'),
        'smri_rpa_file': op.join(basefilename, 'smri_rpa.txt'),
        'smri_lpa_file': op.join(basefilename, 'smri_lpa.txt'),
        'polhemus_nasion_file': op.join(basefilename, 'polhemus_nasion.txt'),
        'polhemus_rpa_file': op.join(basefilename, 'polhemus_rpa.txt'),
        'polhemus_lpa_file': op.join(basefilename, 'polhemus_lpa.txt'),
        'polhemus_headshape_file': op.join(basefilename, 'polhemus_headshape.txt'),
        'forward_model_file': op.join(basefilename, 'forward-fwd.fif'),
        'std_brain': os.environ['FSLDIR'] + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
    }

    return filenames


#############################################################################

def extract_polhemus_from_info(fif_file,
                               outdir,
                               include_eeg_as_headshape=True,
                               include_hpi_as_headshape=True):
    '''  
    # Extract polhemus fids and headshape points from MNE raw.info
    # and write them out in the
    # required file format for rhino (in head/polhemus space in mm)
    # Should only be used with MNE-derived .fif files 
    # that have the expected digitised
    # points held in info['dig'] of fif_file

    Inputs
    ------

    fif_file - string
                Full path to MNE-derived fif file.
    outdir - string
                Full path to directory to write out files to

    Returns
    -------
    
    Polhemus filenames for Rhino to use in call to rhino.coreg

    polhemus_nasion_file : string
    polhemus_rpa_file : string
    polhemus_lpa_file : string
    polhemus_headshape_file : string
    '''

    polhemus_nasion_file = op.join(outdir, 'polhemus_nasion.txt')
    polhemus_rpa_file = op.join(outdir, 'polhemus_rpa.txt')
    polhemus_lpa_file = op.join(outdir, 'polhemus_lpa.txt')
    polhemus_headshape_file = op.join(outdir, 'polhemus_headshape.txt')

    info = read_info(fif_file)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # Setup polhemus files for coreg

    polhemus_headshape_polhemus = []
    for dig in info['dig']:

        # check dig is in HEAD/Polhemus space
        if dig['coord_frame'] != FIFF.FIFFV_COORD_HEAD:
            raise ValueError(
                '{} is not in Head/Polhemus space'.format(dig['ident']))

        if dig['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            if dig['ident'] == FIFF.FIFFV_POINT_LPA:
                polhemus_lpa_polhemus = dig['r']
            elif dig['ident'] == FIFF.FIFFV_POINT_RPA:
                polhemus_rpa_polhemus = dig['r']
            elif dig['ident'] == FIFF.FIFFV_POINT_NASION:
                polhemus_nasion_polhemus = dig['r']
            else:
                raise ValueError('Unknown fiducial: {}'.format(dig['ident']))
        elif dig['kind'] == FIFF.FIFFV_POINT_EXTRA:
            polhemus_headshape_polhemus.append(dig['r'])
        elif dig['kind'] == FIFF.FIFFV_POINT_EEG and include_eeg_as_headshape:
            polhemus_headshape_polhemus.append(dig['r'])
        elif dig['kind'] == FIFF.FIFFV_POINT_HPI and include_hpi_as_headshape:
            polhemus_headshape_polhemus.append(dig['r'])

    np.savetxt(polhemus_nasion_file, polhemus_nasion_polhemus * 1000)
    np.savetxt(polhemus_rpa_file, polhemus_rpa_polhemus * 1000)
    np.savetxt(polhemus_lpa_file, polhemus_lpa_polhemus * 1000)
    np.savetxt(polhemus_headshape_file, np.array(
        polhemus_headshape_polhemus).T * 1000)

    return polhemus_headshape_file, polhemus_nasion_file, \
           polhemus_rpa_file, polhemus_lpa_file


#############################################################################

def compute_surfaces(smri_file,
                     subjects_dir,
                     subject,
                     include_nose=True,
                     cleanup_files=False):
    '''       
    Extracts inner skull, outer skin (scalp) and brain surfaces from 
    passed in smri_file, which is assumed to be a T1, using FSL.
    Assumes that the sMRI file has a valid sform.

    Call get_surfaces_filenames(subjects_dir, subject) to get a file list
    of generated files.

    In more detail:
    1) Transform sMRI to be aligned with the MNI axes so that BET works well
    2) Use bet to skull strip sMRI so that flirt works well
    3) Use flirt to register skull stripped sMRI to MNI space
    4) Use BET/BETSURF to get:
    a) The scalp surface (excluding nose), this gives the sMRI-derived 
        headshape points in native sMRI space, which can be used in the 
        headshape points registration later.
    b) The scalp surface (outer skin), inner skull and brain surface, 
       these can be used for forward modelling later.
       Note that  due to the unusal naming conventions used by BET:
        - bet_inskull_mesh_file is actually the brain surface 
        - bet_outskull_mesh_file is actually the inner skull surface
        - bet_outskin_mesh_file is the outer skin/scalp surface  
    5) Refine scalp outline, adding nose to scalp surface (optional)
    6) Output surfaces in sMRI (native) space and the transform from sMRI
       space to MNI

    
    Inputs
    ------

        smri_file - string
                Full path to structural MRI in niftii format 
                (with .nii.gz extension).
                This is assumed to have a valid sform, i.e. the sform code 
                needs to be 4 or 1, and the sform should transform from voxel
                indices to voxel coords in mm. The axis sform uses to do this
                will be the native/sMRI axis used throughout rhino. The qform 
                will be ignored.

        subjects_dir - string
                Directory to put RHINO subject dirs in.
                Files will be in subjects_dir/subject/rhino/surfaces/
                     
        subject - string
                Subject name dir to put RHINO files in.
                Files will be in subjects_dir/subject/rhino/surfaces/

        include_nose - bool 
                Specifies whether to add the nose to the outer skin
                (scalp) surface. This can help rhino's coreg to work 
                better, assuming that there are headshape points that also 
                include the nose.
                Requires the smri_file to have a FOV that includes the nose!

        cleanup_files - bool
                Specifies whether to cleanup intermediate files in the coreg dir. 
                

    MWW 2021
    '''

    # Note the jargon used varies for xforms and coord spaces, e.g.:
    # MEG (device) -- dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus)-- head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native) -- mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices
    # MRI (native) -- sform (mri2mniaxes) --> MNI axes
    #
    # Rhino does everthing in mm

    print('*** RUNNING OSL RHINO COMPUTE SURFACES ***')

    filenames = get_surfaces_filenames(subjects_dir, subject)

    if include_nose:
        print('The nose is going to be added to the outer skin (scalp) surface.\n\
Please ensure that the structural MRI has a FOV that includes the nose')
    else:
        print('The nose is not going to be added to the outer skin (scalp) surface')

    # Check smri_file
    smri_path, smri_name = op.split(smri_file)
    smri_name, smri_ext2 = op.splitext(smri_name)  # split .gz
    if smri_ext2 != '.gz':
        raise ValueError(
            'smri_file needs to be a niftii file with a .nii.gz extension')

    smri_name, smri_ext1 = op.splitext(smri_name)  # split .nii
    if smri_ext1 != '.nii':
        raise ValueError(
            'smri_file needs to be a niftii file with a .nii.gz extension')

    # Copy smri_name to new file for modification
    copyfile(smri_file, filenames['smri_file'])

    # Rhino will always use the sform, and so we will set the qform to be same
    # as sform for sMRI, to stop the original qform from being used by mistake
    # (e.g. by flirt)
    cmd = 'fslorient -copysform2qform {}'.format(filenames['smri_file'])
    os.system(cmd)

    # We will assume orientation of standard brain is RADIOLOGICAL
    # But let's check that is the case:
    std_orient = rhino_utils._get_orient(filenames['std_brain'])
    if std_orient != 'RADIOLOGICAL':
        raise ValueError('Orientation of standard brain must be RADIOLOGICAL, \
please check output of:\n fslorient -orient {}'.format(filenames['std_brain']))

    # We will assume orientation of sMRI brain is RADIOLOGICAL
    # But let's check that is the case:
    smri_orient = rhino_utils._get_orient(filenames['smri_file'])

    if smri_orient != 'RADIOLOGICAL' and smri_orient != 'NEUROLOGICAL':
        raise ValueError('Cannot determine orientation of subject brain, \
please check output of:\n fslorient -orient {}'.format(filenames['smri_file']))

    # if orientation is not RADIOLOGICAL then force it to be RADIOLOGICAL
    if smri_orient != 'RADIOLOGICAL':
        print('reorienting subject brain to be RADIOLOGICAL')
        os.system('fslorient -forceradiological {}'.format(
            filenames['smri_file']))

    ###########################################################################
    # 1) Transform sMRI to be aligned with the MNI axes so that BET works well
    ###########################################################################        

    # We will start by transforming sMRI
    # so that its voxel indices axes are aligned to MNI's
    # This helps BET work.
    # CALCULATE mri2mniaxes 
    flirt_mri2mniaxes_xform = rhino_utils._get_flirt_xform_between_axes(
        filenames['smri_file'], filenames['std_brain'])

    # Write xform to disk so flirt can use it
    flirt_mri2mniaxes_xform_file = op.join(filenames['basefilename'],
                                           'flirt_mri2mniaxes_xform.txt')
    np.savetxt(flirt_mri2mniaxes_xform_file, flirt_mri2mniaxes_xform)

    # Apply mri2mniaxes xform to smri to get smri_mniaxes, which means sMRIs
    # voxel indices axes are aligned to be the same as MNI's 
    flirt_smri_mniaxes_file = op.join(filenames['basefilename'], \
                                      'flirt_smri_mniaxes.nii.gz')
    os.system('flirt -in {} -ref {} -applyxfm -init {} -out {}'.format(
        filenames['smri_file'],
        filenames['std_brain'],
        flirt_mri2mniaxes_xform_file,
        flirt_smri_mniaxes_file))

    ###########################################################################
    # 2) Use BET to skull strip sMRI so that flirt works well
    ###########################################################################        

    print('Running BET pre-FLIRT...')

    flirt_smri_mniaxes_bet_file = op.join(filenames['basefilename'], \
                                          'flirt_smri_mniaxes_bet')
    os.system('bet2 {} {}'.format(
        flirt_smri_mniaxes_file,
        flirt_smri_mniaxes_bet_file))

    ###########################################################################
    # 3) Use flirt to register skull stripped sMRI to MNI space
    ###########################################################################        

    print('Running FLIRT...')

    # Flirt is run on the skull stripped brains to register the smri_mniaxes
    # to the MNI standard brain

    flirt_mniaxes2mni_file = op.join(filenames['basefilename'], \
                                     'flirt_mniaxes2mni.txt')
    flirt_smri_mni_bet_file = op.join(filenames['basefilename'], \
                                      'flirt_smri_mni_bet.nii.gz')
    os.system('flirt -in {} -ref {} -omat {} -o {}'.format(
        flirt_smri_mniaxes_bet_file,
        filenames['std_brain'],
        flirt_mniaxes2mni_file,
        flirt_smri_mni_bet_file))

    # Calculate overall transform, flirt_mri2mni_xform_file, from smri to MNI
    flirt_mri2mni_xform_file = op.join(filenames['basefilename'], \
                                       'flirt_mri2mni_xform.txt')
    os.system('convert_xfm -omat {} -concat {} {}'.format(
        flirt_mri2mni_xform_file,
        flirt_mniaxes2mni_file,
        flirt_mri2mniaxes_xform_file))

    # and calculate its inverse, flirt_mni2mri_xform_file, from MNI to smri
    flirt_mni2mri_xform_file = op.join(filenames['basefilename'], \
                                       'flirt_mni2mri_xform_file.txt')
    os.system('convert_xfm -omat {}  -inverse {}'.format(
        flirt_mni2mri_xform_file,
        flirt_mri2mni_xform_file))

    # move full smri into MNI space to do full bet and betsurf
    flirt_smri_mni_file = op.join(filenames['basefilename'], \
                                  'flirt_smri_mni.nii.gz')
    os.system('flirt -in {} -ref {} -applyxfm -init {} -out {}'.format(
        filenames['smri_file'],
        filenames['std_brain'],
        flirt_mri2mni_xform_file,
        flirt_smri_mni_file))

    ###########################################################################
    # 4) Use BET/BETSURF to get:
    # a) The scalp surface (excluding nose), this gives the sMRI-derived 
    #     headshape points in native sMRI space, which can be used in the 
    #     headshape points registration later.
    # b) The scalp surface (outer skin), inner skull and brain surface, 
    #    these can be used for forward modelling later.
    #    Note that  due to the unusal naming conventions used by BET:
    #     - bet_inskull_mesh_file is actually the brain surface 
    #     - bet_outskull_mesh_file is actually the inner skull surface
    #     - bet_outskin_mesh_file is the outer skin/scalp surface    
    ###########################################################################        

    # Run BET on smri to get the surface mesh (in MNI space),
    # as BETSURF needs this.

    print('Running BET pre-BETSURF...')

    flirt_smri_mni_bet_file = op.join(filenames['basefilename'], \
                                      'flirt_smri_mni_bet')
    os.system('bet2 {} {} --mesh'.format(
        flirt_smri_mni_file,
        flirt_smri_mni_bet_file))

    ## Run BETSURF - to get the head surfaces in MNI space

    print('Running BETSURF...')

    # Need to provide BETSURF with transform to MNI space.
    # Since flirt_smri_mni_file is already in MNI space,
    # this will just be the identity matrix

    flirt_identity_xform_file = op.join(filenames['basefilename'], \
                                        'flirt_identity_xform.txt')
    np.savetxt(flirt_identity_xform_file, np.eye(4))

    bet_mesh_file = op.join(flirt_smri_mni_bet_file + '_mesh.vtk')
    os.system('betsurf --t1only -o {} {} {} {}'.format(
        flirt_smri_mni_file,
        bet_mesh_file,
        flirt_identity_xform_file,
        op.join(filenames['basefilename'], 'flirt')))

    ###########################################################################
    # 5) Refine scalp outline, adding nose to scalp surface (optional)
    ###########################################################################        

    print('Refining scalp surface...')

    # We do this in MNI big FOV space, to allow the full nose to be included

    # Calculate flirt_mni2mnibigfov_xform 
    mni2mnibigfov_xform = rhino_utils._get_flirt_xform_between_axes(
        from_nii=flirt_smri_mni_file,
        target_nii=filenames['std_brain_bigfov'])

    flirt_mni2mnibigfov_xform_file = op.join(filenames['basefilename'], \
                                             'flirt_mni2mnibigfov_xform.txt')
    np.savetxt(flirt_mni2mnibigfov_xform_file, mni2mnibigfov_xform)

    # Calculate overall transform, from smri to MNI big fov        
    flirt_mri2mnibigfov_xform_file = op.join(filenames['basefilename'], \
                                             'flirt_mri2mnibigfov_xform.txt')
    os.system('convert_xfm -omat {} -concat {} {}'.format(
        flirt_mri2mnibigfov_xform_file,
        flirt_mni2mnibigfov_xform_file,
        flirt_mri2mni_xform_file))

    # move MRI to MNI big FOV space and load in
    flirt_smri_mni_bigfov_file = op.join(filenames['basefilename'], \
                                         'flirt_smri_mni_bigfov')
    os.system('flirt -in {} -ref {} -applyxfm -init {} -out {}'.format(
        filenames['smri_file'],
        filenames['std_brain_bigfov'],
        flirt_mri2mnibigfov_xform_file,
        flirt_smri_mni_bigfov_file))
    vol = nib.load(flirt_smri_mni_bigfov_file + '.nii.gz')

    # move scalp to MNI big FOV space and load in
    flirt_outskin_file = op.join(filenames['basefilename'], 'flirt_outskin_mesh')
    flirt_outskin_bigfov_file = op.join(filenames['basefilename'], \
                                        'flirt_outskin_mesh_bigfov')
    os.system('flirt -in {} -ref {} -applyxfm -init {} -out {}'.format(
        flirt_outskin_file,
        filenames['std_brain_bigfov'],
        flirt_mni2mnibigfov_xform_file,
        flirt_outskin_bigfov_file))
    scalp = nib.load(flirt_outskin_bigfov_file + '.nii.gz')

    # CREATE MASK BY FILLING OUTLINE

    # add a border of ones to the mask, in case the complete head is not in the
    # FOV, without this binary_fill_holes will not work 
    mask = np.ones(np.add(scalp.shape, 2))
    # note that z=100 is where the standard MNI FOV starts in the big FOV
    mask[1:-1, 1:-1, 102:-1] = scalp.get_fdata()[:, :, 101:]
    mask[:, :, :101] = 0

    # We assume that the top of the head is not cutoff by the FOV,
    # we need to assume this so that binary_fill_holes works:
    mask[:, :, -1] = 0

    mask = morphology.binary_fill_holes(mask)

    # remove added border
    mask[:, :, :102] = 0
    mask = mask[1:-1, 1:-1, 1:-1]

    # fig = plt.figure(frameon=False)
    # plt.imshow(mask[99, :, :]); plt.show()

    if include_nose:
        print('Adding nose to scalp surface...')

        # RECLASSIFY BRIGHT VOXELS OUTSIDE OF MASK (TO PUT NOSE INSIDE
        # THE MASK SINCE BET WILL HAVE EXCLUDED IT)

        vol_data = vol.get_fdata()

        # normalise vol data
        vol_data = vol_data / np.max(vol_data.flatten())

        # estimate observation model params of 2 class GMM with diagonal
        # cov matrix where the two classes correspond to inside and
        # outside the bet mask
        means = np.zeros([2, 1])
        means[0] = np.mean(vol_data[np.where(mask == 0)])
        means[1] = np.mean(vol_data[np.where(mask == 1)])
        precisions = np.zeros([2, 1])
        precisions[0] = 1 / np.var(vol_data[np.where(mask == 0)])
        precisions[1] = 1 / np.var(vol_data[np.where(mask == 1)])
        weights = np.zeros([2])
        weights[0] = np.sum((mask == 0))
        weights[1] = np.sum((mask == 1))

        # Create GMM with those observation models
        gm = GaussianMixture(n_components=2, random_state=0,
                             covariance_type="diag")
        gm.means_ = means
        gm.precisions_ = precisions
        gm.precisions_cholesky_ = np.sqrt(precisions)
        gm.weights_ = weights

        # classify voxels outside BET mask with GMM
        labels = gm.predict(vol_data[np.where(mask == 0)].reshape(-1, 1))

        # insert new labels for voxels outside BET mask into mask
        mask[np.where(mask == 0)] = labels

        # ignore anything that is well below the nose and above top of head
        mask[:, :, 0:50] = 0
        mask[:, :, 300:] = 0

        # CLEAN UP MASK
        mask[:, :, 50:300] = morphology.binary_fill_holes(mask[:, :, 50:300])
        mask[:, :, 50:300] = rhino_utils._binary_majority3d(mask[:, :, 50:300])
        mask[:, :, 50:300] = morphology.binary_fill_holes(mask[:, :, 50:300])

        for i in range(mask.shape[0]):
            mask[i, :, 50:300] = morphology.binary_fill_holes(mask[i, :, 50:300])
        for i in range(mask.shape[1]):
            mask[:, i, 50:300] = morphology.binary_fill_holes(mask[:, i, 50:300])
        for i in range(50, 300, 1):
            mask[:, :, i] = morphology.binary_fill_holes(mask[:, :, i])

    # end if include_nose

    # EXTRACT OUTLINE
    outline = np.zeros(mask.shape)
    kernel = getStructuringElement(MORPH_RECT, (3, 3))
    mask = mask.astype(np.uint8)

    # import pdb; pdb.pdb.set_trace()

    # Use morph gradient to find the outline of the solid mask
    for i in range(outline.shape[0]):
        outline[i, :, :] += morphologyEx(mask[i, :, :],
                                         MORPH_GRADIENT, kernel)
    for i in range(outline.shape[1]):
        outline[:, i, :] += morphologyEx(mask[:, i, :],
                                         MORPH_GRADIENT, kernel)
    for i in range(50, 300, 1):
        outline[:, :, i] += morphologyEx(mask[:, :, i],
                                         MORPH_GRADIENT, kernel)
    outline /= 3

    outline[np.where(outline > 0.6)] = 1
    outline[np.where(outline <= 0.6)] = 0
    outline = outline.astype(np.uint8)

    if False:
        fig = plt.figure(frameon=False)
        plt.imshow(vol.get_fdata()[99, :, :])
        plt.show()
        fig = plt.figure(frameon=False)
        plt.imshow(scalp.get_fdata()[99, :, :])
        plt.show()
        fig = plt.figure(frameon=False)
        plt.imshow(mask[99, :, :])
        plt.show()
        fig = plt.figure(frameon=False)
        plt.imshow(outline[99, :, :])
        plt.show()

    # SAVE AS NIFTI
    mesh_name = 'outskin_mesh'
    outline_nii = nib.Nifti1Image(outline, scalp.affine)

    nib.save(outline_nii,
             op.join(flirt_outskin_bigfov_file + '_plus_nose.nii.gz'))

    os.system('fslcpgeom {} {}'.format(
        op.join(flirt_outskin_bigfov_file + '.nii.gz'),
        op.join(flirt_outskin_bigfov_file + '_plus_nose.nii.gz')))

    ## Transform outskin plus nose nii mesh from MNI big FOV to MRI space

    # first we need to invert the flirt_mri2mnibigfov_xform_file xform:
    flirt_mnibigfov2mri_xform_file = op.join(filenames['basefilename'], \
                                             'flirt_mnibigfov2mri_xform.txt')
    os.system('convert_xfm -omat {} -inverse {}'.format(
        flirt_mnibigfov2mri_xform_file,
        flirt_mri2mnibigfov_xform_file))

    os.system('flirt -in {} -ref {} -applyxfm -init {} -out {}'.format(
        op.join(flirt_outskin_bigfov_file + '_plus_nose.nii.gz'),
        filenames['smri_file'],
        flirt_mnibigfov2mri_xform_file,
        filenames['bet_outskin_plus_nose_mesh_file']))

    ###########################################################################
    # 6) Output surfaces in sMRI (native) space and the transform from sMRI
    #    space to MNI
    ###########################################################################

    # Move BET surface to native sMRI space. They are currently in MNI space

    flirt_mni2mri = np.loadtxt(flirt_mni2mri_xform_file)

    xform_mni2mri = rhino_utils._get_xform_from_flirt_xform(
        flirt_mni2mri,
        filenames['std_brain'],
        filenames['smri_file'])

    mni_mri_t = Transform('mni_tal', 'mri', xform_mni2mri)
    write_trans(filenames['mni_mri_t_file'], mni_mri_t)

    # Transform betsurf mask/mesh output from MNI to sMRI space
    for mesh_name in {'outskin_mesh', 'inskull_mesh', 'outskull_mesh'}:
        # xform mask
        os.system('flirt -interp nearestneighbour -in {} -ref {} -applyxfm -init {} -out {}'.format(
            op.join(filenames['basefilename'], 'flirt_' + mesh_name + '.nii.gz'),
            filenames['smri_file'],
            flirt_mni2mri_xform_file,
            op.join(filenames['basefilename'], mesh_name)))

        # xform vtk mesh
        rhino_utils._transform_vtk_mesh(
            op.join(filenames['basefilename'], 'flirt_' + mesh_name + '.vtk'),
            op.join(filenames['basefilename'], 'flirt_' + mesh_name + '.nii.gz'),
            op.join(filenames['basefilename'], mesh_name + '.vtk'),
            op.join(filenames['basefilename'], mesh_name + '.nii.gz'),
            filenames['mni_mri_t_file'])


    ###########################################################################
    # Clean up
    ###########################################################################

    os.system('cp -f {} {}'.format(
        flirt_mni2mri_xform_file,
        filenames['mni2mri_flirt_file']))

    if cleanup_files:
        # CLEAN UP FILES ON DISK
        os.system('rm -f {}'.format(op.join(filenames['basefilename'], 'flirt*')))

    print('*** OSL RHINO COMPUTE SURFACES COMPLETE ***')


#############################################################################

def surfaces_display(subjects_dir, subject):
    '''
    Displays the surfaces extracted from the sMRI using rhino.compute_surfaces
    Display is shown in sMRI (native) space

    Inputs
    ------

        subjects_dir - string
                Directory to put RHINO subject dirs in.
                Files will be in subjects_dir/subject/rhino/surfaces/                        
        subject - string
                Subject name dir to put RHINO files in.
                Files will be in subjects_dir/subject/rhino/surfaces/
    
    Note that bet_inskull_mesh_file is actually the brain surface and
    bet_outskull_mesh_file is the inner skull surface, due to the naming 
    conventions used by BET

    '''

    filenames = get_surfaces_filenames(subjects_dir, subject)

    os.system('fsleyes {} {} {} {} {} &'
              .format(filenames['smri_file'],
                      filenames['bet_inskull_mesh_file'],
                      filenames['bet_outskin_mesh_file'],
                      filenames['bet_outskull_mesh_file'],
                      filenames['bet_outskin_plus_nose_mesh_file']))


#############################################################################

def coreg(fif_file,
          subjects_dir, subject,
          polhemus_headshape_file,
          polhemus_nasion_file, polhemus_rpa_file, polhemus_lpa_file,
          use_headshape=True,
          use_nose=True,
          use_dev_ctf_t=True):
    '''       
    Calculates a linear, affine transform from native sMRI space
    to polhemus (head) space, using headshape points that include the nose
    (if useheadshape = True).
    
    Requires rhino.compute_surfaces to have been run.
    
    This is based on the OSL Matlab version of RHINO.

    Call get_coreg_filenames(subjects_dir, subject) to get a file list
    of generated files.
    
    RHINO firsts registers the polhemus-derived fiducials (nasion, rpa, lpa) 
    in polhemus space to the sMRI-derived fiducials in native sMRI space.

    RHINO then refines this by making use of polhemus-derived headshape points 
    that trace out the surface of the head (scalp), and ideally include  
    the nose.

    Finally, these polhemus-derived headshape points in polhemus space are 
    registered to the sMRI-derived scalp surface in native sMRI space.

    In more detail:
    
    1) Map location of fiducials in MNI standard space brain to native sMRI 
    space. These are then used as the location of the sMRI-derived fiducials
    in native sMRI space.
    2) We have polhemus-derived fids in polhemus space and sMRI-derived fids 
    in native sMRI space. We use these to estimate the affine xform from 
    native sMRI space to polhemus (head) space.
    3) We have the polhemus-derived headshape points in polhemus 
    space and the sMRI-derived headshape (scalp surface) in native sMRI space.  
    We use these to estimate the affine xform from native sMRI space using the 
    ICP algorithm initilaised using the xform estimate in step 2.

    Inputs
    ------
        fif_file - string
                Full path to MNE-derived fif file.

        subjects_dir - string
                Directory to put RHINO subject dirs in.
                Files will be in subjects_dir/subject/rhino/coreg/                        
        subject - string
                Subject name dir to put RHINO files in.
                Files will be in subjects_dir/subject/rhino/coreg/
                
        polhemus_headshape_file - string
                3 x num_pnts space-separated text file of 
                the polhemus derived headshape points in polhemus space in mm.

        polhemus_nasion_file - string 
                3 x 1 text file of the polhemus 
                derived nasion point in polhemus space in mm.
        polhemus_rpa_file - string
                3 x 1 text file of the polhemus 
                derived rpa point in polhemus space in mm.
        polhemus_lpa_file - string
                3 x 1 text file of the polhemus 
                derived lpa point in polhemus space in mm .

        use_headshape - bool 
                Determines whether polhemus derived headshape points are used.
        use_nose - bool 
                Determines whether nose is used to aid coreg, only relevant if
                useheadshape=True
                
        use_dev_ctf_t - bool
                Determines whether to set dev_head_t equal to dev_ctf_t 
                in fif_file's info. This option is only potentially 
                needed for fif files originating from CTF scanners. Will be
                ignored if dev_ctf_t does not exist in info (e.g. if the data
                is from a MEGIN scanner)

    MWW 2021
    '''

    # Note the jargon used varies for xforms and coord spaces:
    # MEG (device) -- dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus)-- head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native) -- mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices
    #
    # Rhino does everthing in mm

    print('*** RUNNING OSL RHINO COREGISTRATION ***')

    filenames = get_coreg_filenames(subjects_dir, subject)
    surfaces_filenames = get_surfaces_filenames(subjects_dir, subject)

    if use_headshape:
        if use_nose:
            print('The MRI-derived nose is going to be used to aid coreg.\n\
Please ensure that rhino.compute_surfaces was run with include_nose=True. \n\
Please ensure that the polhemus headshape points include the nose. \n')
        else:
            print('The MRI-derived nose is not going to be used to aid coreg.\n\
Please ensure that the polhemus headshape points do not include the nose')

    # Copy passed in polhemus pnts
    for fil in ('polhemus_headshape_file',
                'polhemus_nasion_file', 'polhemus_rpa_file', 'polhemus_lpa_file'):
        copyfile(locals()[fil], filenames[fil])

    # Load in the "polhemus-derived fiducial points"
    polhemus_nasion_polhemus = np.loadtxt(polhemus_nasion_file)
    polhemus_rpa_polhemus = np.loadtxt(polhemus_rpa_file)
    polhemus_lpa_polhemus = np.loadtxt(polhemus_lpa_file)
    polhemus_headshape_polhemus = np.loadtxt(polhemus_headshape_file)

    # Load in outskin_mesh_file to get the "sMRI-derived headshape points"
    if use_nose:
        outskin_mesh_file = surfaces_filenames['bet_outskin_plus_nose_mesh_file']
    else:
        outskin_mesh_file = surfaces_filenames['bet_outskin_mesh_file']

    smri_headshape_nativeindex = rhino_utils.niimask2indexpointcloud(outskin_mesh_file)

    ###########################################################################
    # Copy fif_file to new file for modification, and (optionally) changes 
    # dev_head_t to equal dev_ctf_t in fif file info
    ###########################################################################

    if fif_file[-7:] == 'raw.fif':
        raw = read_raw(fif_file)
    elif fif_file[-10:] == 'epochs.fif':
        raw = read_epochs(fif_file)
    else:
        raise ValueError('Invalid fif file, needs to be a *raw.fif or a *epochs.fif file')

    if use_dev_ctf_t:
        dev_ctf_t = raw.info['dev_ctf_t']

        if dev_ctf_t != None:
            print('CTF data')
            print('Setting dev_head_t equal to dev_ctf_t in fif file info. \n\
To turn this off, set use_dev_ctf_t=False')

            dev_head_t, _ = _get_trans(raw.info['dev_head_t'], 'meg', 'head')

            dev_head_t['trans'] = dev_ctf_t['trans']

    raw.save(filenames['fif_file'], overwrite=True)
    fif_file = filenames['fif_file']

    ###########################################################################
    # 1) Map location of fiducials in MNI standard space brain to native sMRI 
    # space. These are then used as the location of the sMRI-derived fiducials
    # in native sMRI space.
    ###########################################################################

    # Known locations of MNI derived fiducials in MNI coords in mm
    mni_nasion_mni = np.asarray([1, 85, -41])
    mni_rpa_mni = np.asarray([83, -20, -65])
    mni_lpa_mni = np.asarray([-83, -20, -65])

    mni_mri_t = read_trans(surfaces_filenames['mni_mri_t_file'])

    # Apply this xform to the mni fids to get what we call the "sMRI-derived 
    # fids" in native space
    smri_nasion_native = rhino_utils.xform_points(
        mni_mri_t['trans'], mni_nasion_mni)
    smri_lpa_native = rhino_utils.xform_points(
        mni_mri_t['trans'], mni_lpa_mni)
    smri_rpa_native = rhino_utils.xform_points(
        mni_mri_t['trans'], mni_rpa_mni)

    ###########################################################################
    # 2) We have polhemus-derived fids in polhemus space and sMRI-derived fids 
    # in native sMRI space. We use these to estimate the affine xform from 
    # native sMRI space to polhemus (head) space.
    ###########################################################################

    # Note that smri_fid_native are the sMRI-derived fids in native space
    polhemus_fid_polhemus = np.concatenate((
        np.reshape(polhemus_nasion_polhemus, [-1, 1]),
        np.reshape(polhemus_rpa_polhemus, [-1, 1]),
        np.reshape(polhemus_lpa_polhemus, [-1, 1])), axis=1)
    smri_fid_native = np.concatenate((
        np.reshape(smri_nasion_native, [-1, 1]),
        np.reshape(smri_rpa_native, [-1, 1]),
        np.reshape(smri_lpa_native, [-1, 1])), axis=1)

    # Estimate the affine xform from native sMRI space to polhemus (head) space
    xform_native2polhemus = rhino_utils.rigid_transform_3D(
        polhemus_fid_polhemus, smri_fid_native)

    smri_fid_polhemus = rhino_utils.xform_points(xform_native2polhemus,
                                                 smri_fid_native)

    ## Now we can transform sMRI-derived headshape pnts into polhemus space:

    # get native (mri) voxel index to native (mri) transform
    xform_nativeindex2native = rhino_utils._get_sform(outskin_mesh_file)['trans']

    # put sMRI-derived headshape points into native space (in mm)
    smri_headshape_native = rhino_utils.xform_points(xform_nativeindex2native,
                                                     smri_headshape_nativeindex)

    # put sMRI-derived headshape points into polhemus space
    smri_headshape_polhemus = rhino_utils.xform_points(xform_native2polhemus,
                                                       smri_headshape_native)

    ###########################################################################
    # 3) We have the polhemus-derived headshape points in polhemus 
    # space and the sMRI-derived headshape (scalp surface) in native sMRI space.  
    # We use these to estimate the affine xform from native sMRI space using the 
    # ICP algorithm initilaised using the xform estimate in step 2.
    ###########################################################################

    if use_headshape:
        print('Running ICP...')

        # Run ICP with multiple initialisations to refine registration of 
        # sMRI-derived headshape points to polhemus derived headshape points, 
        # with both in polhemus space

        # Combined polhemus-derived headshape points and polhemus-derived fids, 
        # with them both in polhemus space
        # These are the "source" points that will be moved around
        polhemus_headshape_polhemus_4icp = np.concatenate(
            (polhemus_headshape_polhemus, polhemus_fid_polhemus), axis=1)

        # import pdb; pdb.pdb.set_trace()
        xform_icp, err, e = rhino_utils.rhino_icp(smri_headshape_polhemus,
                                                  polhemus_headshape_polhemus_4icp,
                                                  30)
        # print((xform_icp*10).astype(int)/10)

    else:
        # No refinement by ICP:
        xform_icp = np.eye(4)

    # Put sMRI-derived headshape points into ICP "refined" polhemus space
    xform_native2polhemus_refined = np.linalg.inv(
        xform_icp) @ xform_native2polhemus
    smri_headshape_polhemus = rhino_utils.xform_points(xform_native2polhemus_refined,
                                                       smri_headshape_native)

    # put sMRI-derived fiducials into refined polhemus space
    smri_nasion_polhemus = rhino_utils.xform_points(xform_native2polhemus_refined,
                                                    smri_nasion_native)
    smri_rpa_polhemus = rhino_utils.xform_points(xform_native2polhemus_refined,
                                                 smri_rpa_native)
    smri_lpa_polhemus = rhino_utils.xform_points(xform_native2polhemus_refined,
                                                 smri_lpa_native)

    ###########################################################################
    # Save coreg info
    ###########################################################################

    # save xforms in MNE format in mm
    xform_native2polhemus_refined_copy = np.copy(
        xform_native2polhemus_refined)

    head_mri_t = Transform('head', 'mri',
                           np.linalg.inv(xform_native2polhemus_refined_copy))
    write_trans(filenames['head_mri_t_file'], head_mri_t)

    nativeindex_native_t = np.copy(xform_nativeindex2native)
    mrivoxel_mri_t = Transform(
        'mri_voxel', 'mri', nativeindex_native_t)
    write_trans(filenames['mrivoxel_mri_t_file'], mrivoxel_mri_t)

    # save sMRI derived fids in mm in polhemus space
    np.savetxt(filenames['smri_nasion_file'], smri_nasion_polhemus)
    np.savetxt(filenames['smri_rpa_file'], smri_rpa_polhemus)
    np.savetxt(filenames['smri_lpa_file'], smri_lpa_polhemus)

    if False:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(smri_headshape_polhemus[0, 0:-1:50],
                   smri_headshape_polhemus[1, 0:-1:50],
                   smri_headshape_polhemus[2, 0:-1:50],
                   'blue', marker='.', s=2)

        ax.scatter(polhemus_headshape_polhemus[0, :],
                   polhemus_headshape_polhemus[1, :],
                   polhemus_headshape_polhemus[2, :],
                   'red', marker='o', s=5)
        plt.show()

    ###########################################################################
    # Create sMRI-derived surfaces in native/mri space in mm,
    # for use by forward modelling
    ###########################################################################

    rhino_utils.create_freesurfer_mesh(infile=surfaces_filenames['bet_inskull_mesh_vtk_file'],
                                       surf_outfile=surfaces_filenames['bet_inskull_surf_file'],
                                       nii_mesh_file=surfaces_filenames['bet_inskull_mesh_file'],
                                       xform_mri_voxel2mri=mrivoxel_mri_t['trans'],
                                       overwrite=True)

    rhino_utils.create_freesurfer_mesh(infile=surfaces_filenames['bet_outskull_mesh_vtk_file'],
                                       surf_outfile=surfaces_filenames['bet_outskull_surf_file'],
                                       nii_mesh_file=surfaces_filenames['bet_outskull_mesh_file'],
                                       xform_mri_voxel2mri=mrivoxel_mri_t['trans'],
                                       overwrite=True)

    rhino_utils.create_freesurfer_mesh(infile=surfaces_filenames['bet_outskin_mesh_vtk_file'],
                                       surf_outfile=surfaces_filenames['bet_outskin_surf_file'],
                                       nii_mesh_file=surfaces_filenames['bet_outskin_mesh_file'],
                                       xform_mri_voxel2mri=mrivoxel_mri_t['trans'],
                                       overwrite=True)

    print('*** OSL RHINO COREGISTRATION COMPLETE ***')


#############################################################################
def coreg_display(subjects_dir, subject,
                  plot_type='surf',
                  display_outskin_with_nose=False,
                  display_sensors=True):
    '''
    Displays the coregistered RHINO scalp surface and polhemus/sensor locations

    Display is done in MEG (device) space (in mm).

    Purple dots are the polhemus derived fiducials (these only get used to
    initialse the coreg, if headshape points are being used).

    Yellow diamonds are the MNI standard space derived fiducials (these are the
    ones that matter)

    Inputs
    ------

        subjects_dir - string
                Directory to put RHINO subject dirs in.
                Files will be in subjects_dir/subject/rhino/coreg/                        
        subject - string
                Subject name dir to put RHINO files in.
                Files will be in subjects_dir/subject/rhino/coreg/

        plot_type - string
                Either:
                    'surf' to do a 3D surface plot using surface meshes
                    'scatter' to do a scatter plot using just point clouds

        display_outskin_with_nose - bool
                Whether to include nose with scalp surface in the display

        display_sensors - bool
                Whether to include sensors in the display

    '''

    # Note the jargon used varies for xforms and coord spaces:
    # MEG (device) -- dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus)-- head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native) -- mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices
    #
    # Rhino does everthing in mm

    surfaces_filenames = get_surfaces_filenames(subjects_dir, subject)

    bet_outskin_plus_nose_mesh_file = surfaces_filenames['bet_outskin_plus_nose_mesh_file']
    bet_outskin_plus_nose_surf_file = surfaces_filenames['bet_outskin_plus_nose_surf_file']
    bet_outskin_mesh_file = surfaces_filenames['bet_outskin_mesh_file']
    bet_outskin_mesh_vtk_file = surfaces_filenames['bet_outskin_mesh_vtk_file']
    bet_outskin_surf_file = surfaces_filenames['bet_outskin_surf_file']

    coreg_filenames = get_coreg_filenames(subjects_dir, subject)
    head_mri_t_file = coreg_filenames['head_mri_t_file']
    mrivoxel_mri_t_file = coreg_filenames['mrivoxel_mri_t_file']

    smri_nasion_file = coreg_filenames['smri_nasion_file']
    smri_rpa_file = coreg_filenames['smri_rpa_file']
    smri_lpa_file = coreg_filenames['smri_lpa_file']
    polhemus_nasion_file = coreg_filenames['polhemus_nasion_file']
    polhemus_rpa_file = coreg_filenames['polhemus_rpa_file']
    polhemus_lpa_file = coreg_filenames['polhemus_lpa_file']
    polhemus_headshape_file = coreg_filenames['polhemus_headshape_file']

    fif_file = coreg_filenames['fif_file']

    if display_outskin_with_nose:
        outskin_mesh_file = bet_outskin_plus_nose_mesh_file
        outskin_mesh_4surf_file = bet_outskin_plus_nose_mesh_file
        outskin_surf_file = bet_outskin_plus_nose_surf_file
    else:
        outskin_mesh_file = bet_outskin_mesh_file
        outskin_mesh_4surf_file = bet_outskin_mesh_vtk_file
        outskin_surf_file = bet_outskin_surf_file

    ###########################################################################
    # Setup xforms

    info = read_info(fif_file)

    mrivoxel_mri_t = read_trans(mrivoxel_mri_t_file)

    head_mri_t = read_trans(head_mri_t_file)
    # get meg to head xform in metres from info
    dev_head_t, _ = _get_trans(info['dev_head_t'], 'meg', 'head')

    # Change xform from metres to mm.
    # Note that MNE xform in fif.info assume metres, whereas we want it
    # in mm. To change units on an xform, just need to change the translation
    # part and leave the rotation alone
    dev_head_t['trans'][0:3, -1] = dev_head_t['trans'][0:3, -1] * 1000

    # We are going to display everything in MEG (device) coord frame in mm
    head_trans = invert_transform(dev_head_t)
    meg_trans = Transform('meg', 'meg')
    mri_trans = invert_transform(
        combine_transforms(dev_head_t, head_mri_t, 'meg', 'mri'))

    ###########################################################################
    # Setup fids and headshape points

    # Load, these are in mm
    polhemus_nasion_polhemus = np.loadtxt(polhemus_nasion_file)
    polhemus_rpa_polhemus = np.loadtxt(polhemus_rpa_file)
    polhemus_lpa_polhemus = np.loadtxt(polhemus_lpa_file)
    polhemus_headshape_polhemus = np.loadtxt(polhemus_headshape_file)

    # Move to MEG (device) space
    polhemus_nasion_meg = rhino_utils.xform_points(head_trans['trans'],
                                                   polhemus_nasion_polhemus)
    polhemus_rpa_meg = rhino_utils.xform_points(head_trans['trans'],
                                                polhemus_rpa_polhemus)
    polhemus_lpa_meg = rhino_utils.xform_points(head_trans['trans'],
                                                polhemus_lpa_polhemus)
    polhemus_headshape_meg = rhino_utils.xform_points(head_trans['trans'],
                                                      polhemus_headshape_polhemus)

    # Load sMRI derived fids, these are in mm in polhemus/head space
    smri_nasion_polhemus = np.loadtxt(smri_nasion_file)
    smri_rpa_polhemus = np.loadtxt(smri_rpa_file)
    smri_lpa_polhemus = np.loadtxt(smri_lpa_file)

    # Move to MEG (device) space
    smri_nasion_meg = rhino_utils.xform_points(
        head_trans['trans'], smri_nasion_polhemus)
    smri_rpa_meg = rhino_utils.xform_points(
        head_trans['trans'], smri_rpa_polhemus)
    smri_lpa_meg = rhino_utils.xform_points(
        head_trans['trans'], smri_lpa_polhemus)

    ###########################################################################
    # Setup MEG sensors

    meg_picks = pick_types(info, meg=True, ref_meg=False, exclude=())

    coil_transs = [_loc_to_coil_trans(info['chs'][pick]['loc'])
                   for pick in meg_picks]
    coils = _create_meg_coils([info['chs'][pick] for pick in meg_picks],
                              acc='normal')

    meg_rrs, meg_tris = list(), list()
    offset = 0
    for coil, coil_trans in zip(coils, coil_transs):
        rrs, tris = _sensor_shape(coil)
        rrs = apply_trans(coil_trans, rrs)
        meg_rrs.append(rrs)
        meg_tris.append(tris + offset)
        offset += len(meg_rrs[-1])
    if len(meg_rrs) == 0:
        print('MEG sensors not found. Cannot plot MEG locations.')
    else:
        meg_rrs = apply_trans(meg_trans, np.concatenate(meg_rrs, axis=0))
        meg_tris = np.concatenate(meg_tris, axis=0)

    # convert to mm
    meg_rrs = meg_rrs * 1000

    ###########################################################################
    # Do plots

    ###########################################################################
    if plot_type == 'surf':
        warnings.filterwarnings("ignore", category=Warning)

        # Initialize figure
        renderer = _get_renderer(None, bgcolor=(
            0.5, 0.5, 0.5), size=(800, 800))

        # Polhemus-derived headshape points
        if len(polhemus_headshape_meg.T) > 0:
            polhemus_headshape_megt = polhemus_headshape_meg.T
            color, scale, alpha = (0, 0.7, 0.7), 0.007, 1
            renderer.sphere(center=polhemus_headshape_megt,
                            color=color, scale=scale * 1000,
                            opacity=alpha, backface_culling=True)

        # MRI-derived nasion, rpa, lpa
        if len(smri_nasion_meg.T) > 0:
            color, scale, alpha = (1, 1, 0), 0.09, 1
            for data in [smri_nasion_meg.T, smri_rpa_meg.T, smri_lpa_meg.T]:
                transform = np.eye(4)
                transform[:3, :3] = mri_trans['trans'][:3, :3] * scale * 1000
                # rotate around Z axis 45 deg first
                transform = transform @ rotation(0, 0, np.pi / 4)
                renderer.quiver3d(
                    x=data[:, 0], y=data[:, 1], z=data[:, 2],
                    u=1., v=0., w=0., color=color, mode='oct',
                    scale=scale, opacity=alpha, backface_culling=True,
                    solid_transform=transform)

        # Polhemus-derived nasion, rpa, lpa
        if len(polhemus_nasion_meg.T) > 0:
            color, scale, alpha = (1, 0, 1), 0.012, 1.5
            for data in [polhemus_nasion_meg.T, polhemus_rpa_meg.T, polhemus_lpa_meg.T]:
                renderer.sphere(center=data, color=color, scale=scale * 1000,
                                opacity=alpha, backface_culling=True)

        if display_sensors:
            # Sensors
            if len(meg_rrs) > 0:
                color, alpha = (0., 0.25, 0.5), 0.2
                surf = dict(rr=meg_rrs, tris=meg_tris)
                renderer.surface(surface=surf, color=color,
                                 opacity=alpha, backface_culling=True)

        # sMRI-derived scalp surface
        # if surf file does not exist, then we must create it
        rhino_utils.create_freesurfer_mesh(infile=outskin_mesh_4surf_file,
                                           surf_outfile=outskin_surf_file,
                                           nii_mesh_file=outskin_mesh_file,
                                           xform_mri_voxel2mri=mrivoxel_mri_t['trans'],
                                           overwrite=True)

        coords_native, faces = nib.freesurfer.read_geometry(outskin_surf_file)

        # Move to MEG (device) space
        coords_meg = rhino_utils.xform_points(
            mri_trans['trans'], coords_native.T).T

        surf_smri = dict(rr=coords_meg, tris=faces)

        renderer.surface(surface=surf_smri, color=(1, 0.8, 1),
                         opacity=0.4, backface_culling=False)

        renderer.set_camera(azimuth=90, elevation=90,
                            distance=600, focalpoint=(0., 0., 0.))
        renderer.show()

    ###########################################################################
    elif plot_type == 'scatter':

        ################
        # Setup scalp surface

        # Load in scalp surface
        # And turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
        smri_headshape_nativeindex = rhino_utils.niimask2indexpointcloud(
            outskin_mesh_file)
        # Move from native voxel indices to native space coordinates (in mm)
        smri_headshape_native = rhino_utils.xform_points(
            mrivoxel_mri_t['trans'], smri_headshape_nativeindex)
        # Move to MEG (device) space
        smri_headshape_meg = rhino_utils.xform_points(mri_trans['trans'],
                                                      smri_headshape_native)

        plt.figure()
        ax = plt.axes(projection='3d')

        if display_sensors:
            color, scale, alpha, marker = (0., 0.25, 0.5), 1, 0.1, '.'
            if len(meg_rrs) > 0:
                meg_rrst = meg_rrs.T  # do plot in mm
                ax.scatter(meg_rrst[0, :], meg_rrst[1, :], meg_rrst[2, :],
                           color=color, marker=marker, s=scale, alpha=alpha)

        color, scale, alpha, marker = (0.5, 0.5, 0.5), 1, 0.2, '.'
        if len(smri_headshape_meg) > 0:
            smri_headshape_megt = smri_headshape_meg
            ax.scatter(smri_headshape_megt[0, 0:-1:20],
                       smri_headshape_megt[1, 0:-1:20],
                       smri_headshape_megt[2, 0:-1:20],
                       color=color, marker=marker, s=scale, alpha=alpha)

        color, scale, alpha, marker = (0, 0.7, 0.7), 10, 0.7, 'o'
        if len(polhemus_headshape_meg) > 0:
            polhemus_headshape_megt = polhemus_headshape_meg
            ax.scatter(polhemus_headshape_megt[0, :],
                       polhemus_headshape_megt[1, :],
                       polhemus_headshape_megt[2, :],
                       color=color, marker=marker, s=scale, alpha=alpha)

        if len(smri_nasion_meg) > 0:
            color, scale, alpha, marker = (1, 1, 0), 200, 1, 'd'
            for data in (smri_nasion_meg, smri_rpa_meg, smri_lpa_meg):
                datat = data
                ax.scatter(datat[0, :], datat[1, :], datat[2, :],
                           color=color, marker=marker, s=scale, alpha=alpha)

        if len(polhemus_nasion_meg) > 0:
            color, scale, alpha, marker = (1, 0, 1), 400, 1, '.'
            for data in (polhemus_nasion_meg, polhemus_rpa_meg, polhemus_lpa_meg):
                datat = data
                ax.scatter(datat[0, :], datat[1, :], datat[2, :],
                           color=color, marker=marker, s=scale, alpha=alpha)

        plt.show()
    else:
        raise ValueError('invalid plot_type')

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('ignore', Warning)


#############################################################################
def forward_model(subjects_dir, subject,
                  model='Single Layer',
                  gridstep=8, mindist=4.0, exclude=0.0,
                  eeg=False, meg=True,
                  verbose=False):
    '''
    Compute forward model

    Inputs
    ------
        subjects_dir - string
            Directory to find RHINO subject dirs in.
        subject - string
            Subject name dir to find RHINO files in.

        model - string
            'Single Layer' or 'Three Layer' 
            'Single Layer' to use single layer (brain/cortex)
            'Three Layer' to three layers (scalp, inner skull, brain/cortex)

        gridstep - int
            A grid will be constructed with the spacing given by ``gridstep`` in mm, 
            generating a volume source space.         
        mindist - float
            Exclude points closer than this distance (mm) to the bounding surface.
        exclude - float
            Exclude points closer than this distance (mm) from the center of mass
            of the bounding surface.
            
        eeg - bool
            Whether or not to compute forward model for eeg sensors
        meg - bool
            Whether or not to compute forward model for meg sensors
        
         gridstep=gridstep, mindist=mindist
    '''

    # compute MNE bem solution  
    if model == 'Single Layer':
        conductivity = (0.3,)  # for single layer
    elif model == 'Three Layer':
        conductivity = (0.3, 0.006, 0.3)  # for three layers
    else:
        raise ValueError('{} is an invalid model choice'.format(model))

    vol_src = setup_volume_source_space(subjects_dir, subject,
                                        gridstep=gridstep,
                                        mindist=mindist)

    # The BEM solution requires a BEM model which describes the geometry of the 
    # head the conductivities of the different tissues.
    # See:
    # https://mne.tools/stable/auto_tutorials/forward/30_forward.html#sphx-glr-auto-tutorials-forward-30-forward-py
    # 
    # Note that the BEM does not involve any use of transforms between spaces.
    # The BEM only depends on the head geometry and conductivities. 
    # It is therefore independent from the MEG data and the head position.
    #
    # This will get the surfaces from:  
    # subjects_dir/subject/bem/inner_skull.surf
    #
    # which is where rhino.setup_volume_source_space will have put it.
    model = make_bem_model(subjects_dir=subjects_dir,
                           subject=subject, ico=None,
                           conductivity=conductivity,
                           verbose=verbose)

    bem = make_bem_solution(model)

    fwd = make_fwd_solution(subjects_dir, subject,
                            src=vol_src,
                            ignore_ref=True,
                            bem=bem, eeg=eeg, meg=meg,
                            verbose=verbose)

    fwd_fname = get_coreg_filenames(subjects_dir, subject)['forward_model_file']

    write_forward_solution(fwd_fname, fwd, overwrite=True)


#############################################################################
def bem_display(subjects_dir, subject,
                plot_type='scatter',
                display_outskin_with_nose=True,
                display_sensors=False):
    '''
    Displays the coregistered RHINO scalp surface and inner skull surface.

    Display is done in MEG (device) space (in mm).

    Inputs
    ------
        subjects_dir - string
                Directory to find RHINO subject dirs in.
        subject - string
                Subject name dir to find RHINO files in.

        plot_type -  string
                Either:
                    'surf' to do a 3D surface plot using surface meshes
                    'scatter' to do a scatter plot using just point clouds

        display_outskin_with_nose - bool
                Whether to include nose with scalp surface in the display

        display_sensors - bool
                Whether to include sensor locations in the display

    '''

    # Note the jargon used varies for xforms and coord spaces:
    # MEG (device) -- dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus)-- head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native) -- mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices
    #
    # Rhino does everthing in mm

    surfaces_filenames = get_surfaces_filenames(subjects_dir, subject)

    bet_outskin_plus_nose_mesh_file = surfaces_filenames['bet_outskin_plus_nose_mesh_file']
    bet_outskin_plus_nose_surf_file = surfaces_filenames['bet_outskin_plus_nose_surf_file']
    bet_outskin_mesh_file = surfaces_filenames['bet_outskin_mesh_file']
    bet_outskin_mesh_vtk_file = surfaces_filenames['bet_outskin_mesh_vtk_file']
    bet_outskin_surf_file = surfaces_filenames['bet_outskin_surf_file']
    bet_inskull_mesh_file = surfaces_filenames['bet_inskull_mesh_file']
    bet_inskull_surf_file = surfaces_filenames['bet_inskull_surf_file']

    coreg_filenames = get_coreg_filenames(subjects_dir, subject)
    head_mri_t_file = coreg_filenames['head_mri_t_file']
    mrivoxel_mri_t_file = coreg_filenames['mrivoxel_mri_t_file']

    fif_file = coreg_filenames['fif_file']

    if display_outskin_with_nose:
        outskin_mesh_file = bet_outskin_plus_nose_mesh_file
        outskin_mesh_4surf_file = bet_outskin_plus_nose_mesh_file
        outskin_surf_file = bet_outskin_plus_nose_surf_file
    else:
        outskin_mesh_file = bet_outskin_mesh_file
        outskin_mesh_4surf_file = bet_outskin_mesh_vtk_file
        outskin_surf_file = bet_outskin_surf_file

    fwd_fname = get_coreg_filenames(subjects_dir, subject)['forward_model_file']
    forward = read_forward_solution(fwd_fname)
    src = forward['src']

    ###########################################################################
    # Setup xforms

    info = read_info(fif_file)

    mrivoxel_mri_t = read_trans(mrivoxel_mri_t_file)

    # get meg to head xform in metres from info
    head_mri_t = read_trans(head_mri_t_file)
    dev_head_t, _ = _get_trans(info['dev_head_t'], 'meg', 'head')

    # Change xform from metres to mm.
    # Note that MNE xform in fif.info assume metres, whereas we want it
    # in mm. To change units on an xform, just need to change the translation
    # part and leave the rotation alone
    dev_head_t['trans'][0:3, -1] = dev_head_t['trans'][0:3, -1] * 1000

    # We are going to display everything in MEG (device) coord frame in mm
    meg_trans = Transform('meg', 'meg')
    mri_trans = invert_transform(
        combine_transforms(dev_head_t, head_mri_t, 'meg', 'mri'))
    head_trans = invert_transform(dev_head_t)

    ###########################################################################
    # Setup MEG sensors

    if display_sensors:
        meg_picks = pick_types(info, meg=True, ref_meg=False, exclude=())

        coil_transs = [_loc_to_coil_trans(info['chs'][pick]['loc'])
                       for pick in meg_picks]
        coils = _create_meg_coils([info['chs'][pick] for pick in meg_picks],
                                  acc='normal')

        meg_rrs, meg_tris = list(), list()
        offset = 0
        for coil, coil_trans in zip(coils, coil_transs):
            rrs, tris = _sensor_shape(coil)
            rrs = apply_trans(coil_trans, rrs)
            meg_rrs.append(rrs)
            meg_tris.append(tris + offset)
            offset += len(meg_rrs[-1])
        if len(meg_rrs) == 0:
            print('MEG sensors not found. Cannot plot MEG locations.')
        else:
            meg_rrs = apply_trans(meg_trans, np.concatenate(meg_rrs, axis=0))
            meg_tris = np.concatenate(meg_tris, axis=0)

        # convert to mm
        meg_rrs = meg_rrs * 1000

    ###########################################################################
    # Setup vol source grid points
    if src is not None:
        # stored points are in metres, convert to mm
        src_pnts = src[0]['rr'][src[0]['vertno'], :] * 1000
        # Move from head space to MEG (device) space
        src_pnts = rhino_utils.xform_points(head_trans['trans'], src_pnts.T).T

    print('Number of dipoles={}'.format(src_pnts.shape[0]))

    ###########################################################################
    # Do plots

    ###########################################################################
    if plot_type == 'surf':
        warnings.filterwarnings("ignore", category=Warning)

        # Initialize figure
        renderer = _get_renderer(None, bgcolor=(
            0.5, 0.5, 0.5), size=(800, 800))

        # Sensors
        if display_sensors:
            if len(meg_rrs) > 0:
                color, alpha = (0., 0.25, 0.5), 0.2
                surf = dict(rr=meg_rrs, tris=meg_tris)
                renderer.surface(surface=surf, color=color,
                                 opacity=alpha, backface_culling=True)

        # sMRI-derived scalp surface
        rhino_utils.create_freesurfer_mesh(infile=outskin_mesh_4surf_file,
                                           surf_outfile=outskin_surf_file,
                                           nii_mesh_file=outskin_mesh_file,
                                           xform_mri_voxel2mri=mrivoxel_mri_t['trans'],
                                           overwrite=True)

        coords_native, faces = nib.freesurfer.read_geometry(outskin_surf_file)

        # Move to MEG (device) space
        coords_meg = rhino_utils.xform_points(
            mri_trans['trans'], coords_native.T).T

        surf_smri = dict(rr=coords_meg, tris=faces)

        # plot surface
        renderer.surface(surface=surf_smri, color=(0.85, 0.85, 0.85),
                         opacity=0.3, backface_culling=False)

        # Inner skull surface
        # Load in surface, this is in mm
        coords_native, faces = nib.freesurfer.read_geometry(
            bet_inskull_surf_file)

        # Move to MEG (device) space
        coords_meg = rhino_utils.xform_points(
            mri_trans['trans'], coords_native.T).T

        surf_smri = dict(rr=coords_meg, tris=faces)

        # plot surface
        renderer.surface(surface=surf_smri, color=(0.25, 0.25, 0.25),
                         opacity=0.25, backface_culling=False)

        # vol source grid points
        if src is not None and len(src_pnts.T) > 0:
            color, scale, alpha = (1, 0, 0), 0.001, 1
            renderer.sphere(center=src_pnts, color=color, scale=scale * 1000,
                            opacity=alpha, backface_culling=True)

        renderer.set_camera(azimuth=90, elevation=90,
                            distance=600, focalpoint=(0., 0., 0.))
        renderer.show()

    ###########################################################################
    elif plot_type == 'scatter':

        ################
        # Setup scalp surface

        # Load in scalp surface
        # And turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
        smri_headshape_nativeindex = rhino_utils.niimask2indexpointcloud(
            outskin_mesh_file)
        # Move from native voxel indices to native space coordinates (in mm)
        smri_headshape_native = rhino_utils.xform_points(mrivoxel_mri_t['trans'],
                                                         smri_headshape_nativeindex)
        # Move to MEG (device) space
        smri_headshape_meg = rhino_utils.xform_points(mri_trans['trans'],
                                                      smri_headshape_native)

        ################
        # Setup inner skull surface

        # Load in inner skull surface
        # And turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
        inner_skull_nativeindex = rhino_utils.niimask2indexpointcloud(
            bet_inskull_mesh_file)
        # Move from native voxel indices to native space coordinates (in mm)
        inner_skull_native = rhino_utils.xform_points(mrivoxel_mri_t['trans'],
                                                      inner_skull_nativeindex)
        # Move to MEG (device) space
        inner_skull_meg = rhino_utils.xform_points(mri_trans['trans'],
                                                   inner_skull_native)

        ax = plt.axes(projection='3d')

        # sensors
        if display_sensors:
            color, scale, alpha, marker = (0., 0.25, 0.5), 2, 0.2, '.'
            if len(meg_rrs) > 0:
                meg_rrst = meg_rrs.T  # do plot in mm
                ax.scatter(meg_rrst[0, :], meg_rrst[1, :], meg_rrst[2, :],
                           color=color, marker=marker, s=scale, alpha=alpha)

        # scalp
        color, scale, alpha, marker = (0.75, 0.75, 0.75), 6, 0.2, '.'
        if len(smri_headshape_meg) > 0:
            smri_headshape_megt = smri_headshape_meg
            ax.scatter(smri_headshape_megt[0, 0:-1:20],
                       smri_headshape_megt[1, 0:-1:20],
                       smri_headshape_megt[2, 0:-1:20],
                       color=color, marker=marker, s=scale, alpha=alpha)

        # inner skull
        inner_skull_megt = inner_skull_meg
        color, scale, alpha, marker = (0.5, 0.5, 0.5), 6, 0.2, '.'
        ax.scatter(inner_skull_megt[0, 0:-1:20], inner_skull_megt[1, 0:-1:20],
                   inner_skull_megt[2, 0:-1:20],
                   color=color, marker=marker, s=scale, alpha=alpha)

        # vol source grid points
        if src is not None and len(src_pnts.T) > 0:
            color, scale, alpha, marker = (1, 0, 0), 1, 0.5, '.'
            src_pntst = src_pnts.T
            ax.scatter(src_pntst[0, :], src_pntst[1, :], src_pntst[2, :],
                       color=color, marker=marker, s=scale, alpha=alpha)

        plt.show()
    else:
        raise ValueError('invalid plot_type')

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('ignore', Warning)


#############################################################################
def setup_volume_source_space(subjects_dir, subject,
                              gridstep=5, mindist=5.0, exclude=0.0):
    '''
    Set up a volume source space grid inside the inner skull surface.
    This is a RHINO specific version of mne.setup_volume_source_space.

    Inputs
    ------
    subjects_dir - string
            Directory to find RHINO subject dirs in.
    subject - string
            Subject name dir to find RHINO files in.

    gridstep - int
        A grid will be constructed with the spacing given by ``gridstep`` in mm, 
        generating a volume source space.         
    mindist - float
        Exclude points closer than this distance (mm) to the bounding surface.
    exclude - float
        Exclude points closer than this distance (mm) from the center of mass
        of the bounding surface.

    Returns
    -------

    src - SourceSpaces
        A :class:`SourceSpaces` object containing a single source space.


    See Also
    --------

    mne.setup_volume_source_space

    Notes
    -----

    This is a RHINO specific version of mne.setup_volume_source_space, which 
    can handle smri's that are niftii files. This specifically
    uses the inner skull surface in:
        get_surfaces_filenames(subjects_dir, subject)['bet_inskull_surf_file'] 
    to define the source space grid.

    This will also copy the:
        get_surfaces_filenames(subjects_dir, subject)['bet_inskull_surf_file'] 
    file to: 
        subjects_dir/subject/bem/inner_skull.surf
    since this is where mne expects to find it when mne.make_bem_model 
    is called.

    The coords of points to reconstruct to can be found in the output here: 
        src[0]['rr'][src[0]['vertno']]
    where they are in native MRI space in metres.

    '''

    from mne.surface import read_surface, write_surface
    from mne.source_space import _make_volume_source_space, _complete_vol_src
    from copy import deepcopy

    pos = int(gridstep)

    coreg_filenames = get_coreg_filenames(subjects_dir, subject)
    surfaces_filenames = get_surfaces_filenames(subjects_dir, subject)

    ###########################################################################
    # Move the surfaces to where MNE expects to find them for the 
    # forward modelling, see make_bem_model in mne/bem.py

    # First make sure bem directory exists:
    bem_dir_name = op.join(subjects_dir, subject, 'bem')
    if not os.path.isdir(bem_dir_name):
        os.mkdir(bem_dir_name)

    # Note that due to the unusal naming conventions used by BET and MNE:
    # - bet_inskull_*_file is actually the brain surface 
    # - bet_outskull_*_file is actually the inner skull surface
    # - bet_outskin_*_file is the outer skin/scalp surface  
    # These correspond in mne to (in order):
    # - inner_skull
    # - outer_skull
    # - outer_skin
    # 
    # This means that for single shell model, i.e. with conductivities set
    # to length one, the surface used by MNE willalways be the inner_skull, i.e. 
    # it actually corresponds to the brain/cortex surface!! Not sure that is
    # correct/optimal.
    #
    # Note that this is done in Fieldtrip too!, see the 
    # "Realistic single-shell model, using brain surface from segmented mri"
    # section at:
    # https://www.fieldtriptoolbox.org/example/make_leadfields_using_different_headmodels/#realistic-single-shell-model-using-brain-surface-from-segmented-mri
    # 
    # However, others are clear that it should really be the actual inner surface
    # of the skull, see the "single-shell Boundary Element Model (BEM)" bit at:
    # https://imaging.mrc-cbu.cam.ac.uk/meg/SpmForwardModels
    #
    # To be continued... need to get in touch with mne folks perhaps?

    if True:
        verts, tris = read_surface(surfaces_filenames['bet_inskull_surf_file'])
        tris = tris.astype(int)
        write_surface(op.join(bem_dir_name, 'inner_skull.surf'),
                      verts, tris, file_format='freesurfer', overwrite=True)
        print('Using bet_inskull_surf_file for single shell surface')
    else:
        verts, tris = read_surface(surfaces_filenames['bet_outskull_surf_file'])
        tris = tris.astype(int)
        write_surface(op.join(bem_dir_name, 'inner_skull.surf'),
                      verts, tris, file_format='freesurfer', overwrite=True)
        print('Using bet_outskull_surf_file for single shell surface')

    verts, tris = read_surface(surfaces_filenames['bet_outskull_surf_file'])
    tris = tris.astype(int)
    write_surface(op.join(bem_dir_name, 'outer_skull.surf'),
                  verts, tris, file_format='freesurfer', overwrite=True)

    verts, tris = read_surface(surfaces_filenames['bet_outskin_surf_file'])
    tris = tris.astype(int)
    write_surface(op.join(bem_dir_name, 'outer_skin.surf'),
                  verts, tris, file_format='freesurfer', overwrite=True)

    ###########################################################################
    # Setup main MNE call to _make_volume_source_space

    surface = op.join(subjects_dir, subject, 'bem', 'inner_skull.surf')

    pos = float(pos)
    pos /= 1000.0  # convert pos to m from mm for MNE call

    ###################################################
    def get_mri_info_from_nii(mri, mrivoxel_mri_t_file):
        out = dict()
        dims = nib.load(mri).get_fdata().shape

        out.update(
            mri_width=dims[0], mri_height=dims[1],
            mri_depth=dims[1], mri_volume_name=mri)

        return out

    vol_info = get_mri_info_from_nii(
        surfaces_filenames['smri_file'], coreg_filenames['mrivoxel_mri_t_file'])

    surf = read_surface(surface, return_dict=True)[-1]

    surf = deepcopy(surf)
    surf['rr'] *= 1e-3  # must be in metres for MNE call

    # Main MNE call to _make_volume_source_space
    sp = _make_volume_source_space(
        surf, pos, exclude, mindist, surfaces_filenames['smri_file'], None,
        vol_info=vol_info, single_volume=False)

    sp[0]['type'] = 'vol'

    ###########################################################################
    # Save and return result

    sp = _complete_vol_src(sp, subject)

    # add dummy mri_ras_t and vox_mri_t transforms as these are needed for the
    # forward model to be saved (for some reason)
    sp[0]['mri_ras_t'] = Transform('mri', 'ras')

    sp[0]['vox_mri_t'] = Transform('mri_voxel', 'mri')

    if sp[0]['coord_frame'] != FIFF.FIFFV_COORD_MRI:
        raise RuntimeError('source space is not in MRI coordinates')

    return sp


#############################################################################

def make_fwd_solution(subjects_dir, subject,
                      src,
                      bem,
                      meg=True,
                      eeg=True,
                      mindist=0.0,
                      ignore_ref=False,
                      n_jobs=1,
                      verbose=None):
    '''Calculate a forward solution for a subject. This is a RHINO wrapper
    for mne.make_forward_solution

    Inputs
    ------

    See mne.make_forward_solution for the full set of parameters, with the
    exception of:

    subjects_dir - string
            Directory to find RHINO subject dirs in.
    subject - string
            Subject name dir to find RHINO files in.

    Returns
    -------
    fwd : instance of Forward
        The forward solution.

    Notes
    -----
    Forward modelling is done in head space.

    The coords of points to reconstruct to can be found in the output here: 
        fwd['src'][0]['rr'][fwd['src'][0]['vertno']]
    where they are in head space in metres.

    The same coords of points to reconstruct to can be found in the input here: 
        src[0]['rr'][src[0]['vertno']]
    where they are in native MRI space in metres.
    '''

    fif_file = get_coreg_filenames(subjects_dir, subject)['fif_file']

    # Note, forward model is done in Head space:
    head_mri_trans_file = get_coreg_filenames(subjects_dir, subject)['head_mri_t_file']

    # Src should be in MRI space. Let's just check that is the
    # case
    if src[0]['coord_frame'] != FIFF.FIFFV_COORD_MRI:
        raise RuntimeError('src is not in MRI coordinates')

    # We need the transformation from MRI to HEAD coordinates
    # (or vice versa)
    if isinstance(head_mri_trans_file, str):
        head_mri_t = read_trans(head_mri_trans_file)
    else:
        head_mri_t = head_mri_trans_file

    # RHINO does everything in mm, so need to convert it to metres which is 
    # what MNE expects.
    # To change units on an xform, just need to change the translation
    # part and leave the rotation alone:
    head_mri_t['trans'][0:3, -1] = head_mri_t['trans'][0:3, -1] / 1000

    if isinstance(bem, str):
        bem = read_bem_solution(bem)
    else:
        if not isinstance(bem, ConductorModel):
            raise TypeError('bem must be a string or ConductorModel')

        bem = bem.copy()

    for ii in range(len(bem['surfs'])):
        bem['surfs'][ii]['tris'] = bem['surfs'][ii]['tris'].astype(int)

    info = read_info(fif_file)

    ###########################################################################
    # Main MNE call    
    fwd = make_forward_solution(info, trans=head_mri_t, src=src,
                                bem=bem, eeg=eeg, meg=meg,
                                mindist=mindist, ignore_ref=ignore_ref,
                                n_jobs=n_jobs, verbose=verbose)

    # fwd should be in Head space. Let's just check that is the case:
    if fwd['src'][0]['coord_frame'] != FIFF.FIFFV_COORD_HEAD:
        raise RuntimeError('fwd[\'src\'][0] is not in HEAD coordinates')

    return fwd


#############################################################################

def recon_ts2mri_nii(subjects_dir, subject,
                     recon_timeseries,
                     out_nii_fname,
                     times=None):
    '''
    Writes niftii files to native sMRI space for the passed in recon_timeseries
    
    Inputs
    ------

    subjects_dir - string
            Directory to find RHINO subject dirs in.
    subject - string
            Subject name dir to find RHINO files in.


    recon_timeseries : numpy array
            (nvoxels,) numpy array 
            The coordinates of the
            voxels will by assumed to be those contained in fwd, i.e.
            fwd['src'][0]['rr'][fwd['src']['vertno']]

    out_nii_fname : list of strings | string
            Niftii filename to use for each volume in volumes.
            

    Returns
    -------
    out_nii_fname : list of strings
            Niftii filenames written.

    '''

    if len(recon_timeseries.shape) == 1:
        recon_timeseries = np.reshape(recon_timeseries, [recon_timeseries.shape[0], 1])

    surfaces_filenames = get_surfaces_filenames(subjects_dir, subject)
    coreg_filenames = get_coreg_filenames(subjects_dir, subject)

    smri_file = surfaces_filenames['smri_file']
    head_mri_trans_file = coreg_filenames['head_mri_t_file']

    fwd = read_forward_solution(coreg_filenames['forward_model_file'])

    # fwd should be in Head space. Let's double just check that is the case
    if fwd['src'][0]['coord_frame'] != FIFF.FIFFV_COORD_HEAD:
        raise RuntimeError('fwd[\'src\'][0] is not in HEAD coordinates')

    ##############
    # estimate gridstep from forward model
    rr = fwd['src'][0]['rr']

    store = []
    for ii in range(rr.shape[0]):
        store.append(np.sqrt(np.sum(np.square(rr[ii, :] - rr[0, :]))))
    store = np.asarray(store)
    gridstep = int(np.round(np.min(store[np.where(store > 0)]) * 1000))
    print('Using gridstep = {}mm'.format(gridstep))
    ##############

    ##############
    # Get hold of coords of points reconstructed to.
    # Note, MNE forward model is done in head space in metres.
    # Rhino does everything in mm
    vs = fwd['src'][0]
    coords_head = vs['rr'][vs['vertno']] * 1000  # in mm
    ##############

    in_file = smri_file

    # create background file name for chosen gridstep
    out_gridstep_mm_file = in_file.replace(
        '.nii.gz', '_{}mm.nii.gz'.format(gridstep))

    # Sample smri to the gridstep resolution
    # output is out_gridstep_mm_file
    os.system('flirt -in {} -ref {} -out {} -applyisoxfm {}'.format(
        in_file,
        in_file,
        out_gridstep_mm_file,
        gridstep))

    # get transform from mri to mri voxel indices for smri at gridstep resolution
    mri2mri_vox_t = invert_transform(rhino_utils._get_sform(out_gridstep_mm_file))

    # Compute head to native MRI voxel index xform    
    head_vox2outspace_vox_t = combine_transforms(read_trans(head_mri_trans_file),
                                                 mri2mri_vox_t, 'head', 'mri_voxel')

    # Convert to native voxel index output space
    coords_out_voxel = np.round(apply_trans(
        head_vox2outspace_vox_t, coords_head)).astype(int)

    out_gridstep_mm = nib.load(out_gridstep_mm_file)

    affine = out_gridstep_mm.affine

    if coords_out_voxel.shape[0] != recon_timeseries.shape[0]:
        raise ValueError('coords_out_voxel.shape[0] ~= recon_timeseries[0] \n\
Passed in volumes are not compatible with passed in forward model, fwd.\n\
Each volume should be a (nvoxels,) array')

    vol = nib.load(out_gridstep_mm_file).get_fdata()
    vol = np.zeros(np.append((vol.shape), recon_timeseries.shape[1]))

    for ii in range(coords_out_voxel.shape[0]):
        try:
            vol[coords_out_voxel[ii, 0], coords_out_voxel[ii, 1],
            coords_out_voxel[ii, 2], :] = recon_timeseries[ii, :]
        except IndexError:
            print('Index out of bounds ignored')

    # SAVE AS NIFTI
    vol_nii = nib.Nifti1Image(vol, affine)

    vol_nii.header.set_xyzt_units(2)  # mm
    if times is not None:
        vol_nii.header['pixdim'][4] = times[1] - times[0]
        vol_nii.header['toffset'] = -0.5
        vol_nii.header.set_xyzt_units(2, 8)  # mm and secs

    nib.save(vol_nii, out_nii_fname)

    # os.system('fslcpgeom {} {}'.format(out_gridstep_mm_file, out_nii_fname))

    return out_nii_fname


#############################################################################

def _write_mni_nii(subjects_dir, subject,
                   nii_file_names,
                   mni_resolution=2):
    '''
    Writes niftii files in MNI space at the resolution given by mni_resolution
    (in mm) for each of the passed in nii_file_names, which are assumed to
    be in native MRI space

    Inputs
    ------

    subjects_dir - string
            Directory to find RHINO subject dirs in.
    subject - string
            Subject name dir to find RHINO files in.

    nii_file_names : list of strings
            Niftii filenames to transform to MNI space. These input are 
            assumed to be in native MRI space.
            These will usually have come from calling rhino.write_nii

    mni_resolution : float
            Resolution to write out in MNI space, in mm

    Returns
    -------
    mni_files : list of strings
            Niftii filenames written.
    
    std_mni_brain_file : string
            standard MNI brain at mni_resolution

    '''

    # check nii_file_names to see if it is a list, 
    if type(nii_file_names) is not list:
        # Create a list from the single input
        single_item = nii_file_names
        nii_file_names = []
        nii_file_names.append(single_item)

    surfaces_filenames = get_surfaces_filenames(subjects_dir, subject)
    mni2mri_flirt_file = surfaces_filenames['mni2mri_flirt_file']

    # first we need to invert the mni2mri_flirt xform:
    xform_path, xform_name = op.split(mni2mri_flirt_file)
    mri2mni_flirt_file = op.join(xform_path, 'mri2mni_flirt.txt')

    os.system('convert_xfm -omat {} -inverse {}'.format(
        mri2mni_flirt_file,
        mni2mri_flirt_file))

    mni_files = list()
    for ii in range(len(nii_file_names)):

        if not op.isfile(nii_file_names[ii]):
            raise ValueError('nii_file_names[{}]:\n{}\n does not exist'
                             .format(ii, nii_file_names[ii]))

        nii_path, nii_name = op.split(nii_file_names[ii])
        nii_name, nii_ext2 = op.splitext(nii_name)  # split .gz
        nii_name, nii_ext1 = op.splitext(nii_name)  # split .nii

        mni_file = op.join(nii_path, nii_name +
                           '_mni_{}mm.nii.gz'.format(mni_resolution))

        std_brain = os.environ['FSLDIR'] + \
                    '/data/standard/MNI152_T1_1mm_brain.nii.gz'

        # Sample std_brain to the desired resolution
        # output is std_brain_mm_file
        std_mni_brain_file = op.join(nii_path,
                                     'MNI152_T1_{}mm_brain.nii.gz'.format(mni_resolution))

        # create std brain of the required resolution
        os.system('flirt -in {} -ref {} -out {} -applyisoxfm {}'.format(
            std_brain,
            std_brain,
            std_mni_brain_file,
            mni_resolution))

        # move from native mri to MNI space
        os.system('flirt -in {} -ref {} -applyxfm \
-init {} -out {}'.format(
            nii_file_names[ii],
            std_mni_brain_file,
            mri2mni_flirt_file,
            mni_file))

        # apply MNI brain as mask:
        os.system('fslmaths {} -mas {} {}'.format(
            mni_file,
            std_mni_brain_file,
            mni_file))

        mni_files.append(mni_file)

    return mni_files, std_mni_brain_file


#############################################################################
def get_recon_timeseries(subjects_dir, subject, coord_mni, recon_timeseries_head):
    '''
    Gets the reconstructed time series nearest to the passed in coordinate
    in MNI space
    
    Inputs
    ------

    subjects_dir - string
            Directory to find RHINO subject dirs in.
            
    subject - string
            Subject name dir to find RHINO files in.

    coord_mni : (3,) np.array
            3D coordinate in MNI space to get timeseries for

    recon_timeseries_head : (ndipoles, ntpts) np.array
            Reconstructed time courses in head (polhemus) space
            Assumes that the dipoles are the same (and in the same order)
            as those in the forward model,
            coreg_filenames['forward_model_file']

    Returns
    -------
    The timecourse in recon_timeseries_head nearest to coord_mni
    
    '''

    surfaces_filenames = get_surfaces_filenames(subjects_dir, subject)
    coreg_filenames = get_coreg_filenames(subjects_dir, subject)

    # get coord_mni in mri space
    mni_mri_t = rhino_utils.read_trans(surfaces_filenames['mni_mri_t_file'])
    coord_mri = rhino_utils.xform_points(mni_mri_t['trans'], coord_mni)

    # Get hold of coords of points reconstructed to.
    # Note, MNE forward model is done in head space in metres.
    # Rhino does everything in mm
    fwd = read_forward_solution(coreg_filenames['forward_model_file'])
    vs = fwd['src'][0]
    recon_coords_head = vs['rr'][vs['vertno']] * 1000  # in mm

    # convert coords_head from head to mri space to get index of reconstructed
    # coordinate nearest to coord_mni
    head_mri_t = rhino_utils.read_trans(coreg_filenames['head_mri_t_file'])
    recon_coords_mri = rhino_utils.xform_points(head_mri_t['trans'], recon_coords_head.T).T

    recon_index, d = rhino_utils._closest_node(coord_mri.T, recon_coords_mri)

    recon_timeseries = np.abs(recon_timeseries_head[recon_index, :]).T

    return recon_timeseries


#############################################################################
def resample_recon_ts(subjects_dir, subject,
                      recon_timeseries,
                      spatial_resolution=None,
                      reference_brain='mni'):
    '''
    Spatially resamples a (ndipoles x ntpts) array of reconstructed time 
    courses (in head/polhemus space) to dipoles on the brain
    grid of the specified reference brain 
    
    Inputs
    ------

    subjects_dir - string
            Directory to find RHINO subject dirs in.
            
    subject - string
            Subject name dir to find RHINO files in.
            
    recon_timeseries : (ndipoles, ntpts) np.array
            Reconstructed time courses (in head (polhemus) space).
            Assumes that the dipoles are the same (and in the same order)
            as those in the forward model,
            coreg_filenames['forward_model_file'].
            Typically derive from the VolSourceEstimate's output by
            MNE source recon methods, e.g. mne.beamformer.apply_lcmv, obtained
            using a forward model generated by Rhino.
                                    
    spatial_resolution - int
            Resolution to use for the reference brain in mm 
            (must be an integer, or will be cast to nearest int)
            If None, then the gridstep used in coreg_filenames['forward_model_file']
            is used.
    
    reference_brain - string, 'mni' or 'mri'
            'mni' indicates that the reference_brain is the stdbrain in MNI space
            'mri' indicates that the reference_brain is the sMRI in native/mri space

    Returns
    -------
    recon_timeseries_out - (ndipoles, ntpts) np.array
            Reconstructed time courses resampled on the reference brain grid
   
    reference_brain_fname - string
            File name of the requested reference brain at the requested 
            spatial resolution, int(spatial_resolution)
            (with zero for background, and !=0 for brain)      
            
    '''

    surfaces_filenames = get_surfaces_filenames(subjects_dir, subject)
    coreg_filenames = get_coreg_filenames(subjects_dir, subject)

    #########
    # Get hold of coords of points reconstructed to.
    # Note, MNE forward model is done in head space in metres.
    # Rhino does everything in mm
    fwd = read_forward_solution(coreg_filenames['forward_model_file'])
    vs = fwd['src'][0]
    recon_coords_head = vs['rr'][vs['vertno']] * 1000  # in mm

    ##############
    if spatial_resolution is None:
        # estimate gridstep from forward model
        rr = fwd['src'][0]['rr']

        store = []
        for ii in range(rr.shape[0]):
            store.append(np.sqrt(np.sum(np.square(rr[ii, :] - rr[0, :]))))
        store = np.asarray(store)
        spatial_resolution = int(np.round(np.min(store[np.where(store > 0)]) * 1000))
        print('Using spatial_resolution = {}mm'.format(spatial_resolution))

    spatial_resolution = int(spatial_resolution)

    if reference_brain == 'mni':
        # reference is mni stdbrain

        #########
        # convert recon_coords_head from head to mni space
        head_mri_t = rhino_utils.read_trans(coreg_filenames['head_mri_t_file'])
        recon_coords_mri = rhino_utils.xform_points(head_mri_t['trans'], recon_coords_head.T).T

        mni_mri_t = rhino_utils.read_trans(surfaces_filenames['mni_mri_t_file'])
        recon_coords_out = rhino_utils.xform_points(np.linalg.inv(mni_mri_t['trans']), recon_coords_mri.T).T

        reference_brain = os.environ['FSLDIR'] + \
                          '/data/standard/MNI152_T1_1mm_brain.nii.gz'

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = op.join(coreg_filenames['basefilename'],
                                            'MNI152_T1_{}mm_brain.nii.gz'.format(spatial_resolution))

    elif reference_brain == 'mri':
        # reference is smri

        #########
        # convert recon_coords_head from head to mri space
        head_mri_t = rhino_utils.read_trans(coreg_filenames['head_mri_t_file'])
        recon_coords_out = rhino_utils.xform_points(head_mri_t['trans'], recon_coords_head.T).T

        reference_brain = surfaces_filenames['smri_file']

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = reference_brain.replace(
            '.nii.gz', '_{}mm.nii.gz'.format(spatial_resolution))

    else:
        ValueError('Invalid out_space, should be mni or mri')

    #########
    # get coordinates from reference brain at resolution spatial_resolution

    # create std brain of the required resolution
    os.system('flirt -in {} -ref {} -out {} -applyisoxfm {}'.format(
        reference_brain,
        reference_brain,
        reference_brain_resampled,
        spatial_resolution))

    coords_out_mm, vals = rhino_utils.niimask2mmpointcloud(reference_brain_resampled)

    #########
    # for each coords_mni find nearest coord in recon_coords_out
    recon_timeseries_out = np.zeros([coords_out_mm.shape[1], recon_timeseries.shape[1]])
    for cc in range(coords_out_mm.shape[1]):
        recon_index, dist = rhino_utils._closest_node(coords_out_mm[:, cc], recon_coords_out)

        if dist < spatial_resolution:
            recon_timeseries_out[cc, :] = recon_timeseries[recon_index, :]

    reference_brain_fname = reference_brain_resampled

    return recon_timeseries_out, reference_brain_fname


#############################################################################
def recon_ts2nii(subjects_dir, subject,
                 recon_timeseries,
                 out_nii_fname,
                 spatial_resolution=None,
                 reference_brain='mni',
                 times=None):
    '''
    Converts a (ndipoles,tpts) array of reconstructed timeseries (in 
    head/polhemus space) to the corresponding
    dipoles in a standard brain grid in MNI space and outputs them as a 
    niftii file.

    Inputs
    ------
            
    subjects_dir - string
            Directory to find RHINO subject dirs in.
            
    subject - string
            Subject name dir to find RHINO files in.
            
    recon_timeseries : (ndipoles, ntpts) np.array
            Reconstructed time courses (in head (polhemus) space).
            Assumes that the dipoles are the same (and in the same order)
            as those in the forward model,
            coreg_filenames['forward_model_file'].
            Typically derive from the VolSourceEstimate's output by
            MNE source recon methods, e.g. mne.beamformer.apply_lcmv, obtained
            using a forward model generated by Rhino.
                                    
    spatial_resolution - int
            Resolution to use for the reference brain in mm 
            (must be an integer, or will be cast to nearest int)
            If None, then the gridstep used in coreg_filenames['forward_model_file']
            is used.
            
    reference_brain - string, 'mni' or 'mri'
            'mni' indicates that the reference_brain is the stdbrain in MNI space
            'mri' indicates that the reference_brain is the sMRI in native/mri space
            
    times = (ntpts, ) np.array
            Times points in seconds.
            Will assume that these are regularly spaced
            
    Returns
    -------
    out_nii_fname - string
            Name of output niftii file    
   
    reference_brain_fname - string
            File name of standard brain mask in MNI space at requested resolution,
            int(stdbrain_resolution)
            (with zero for background, and !=0 for the mask)      
            

    '''

    if len(recon_timeseries.shape) == 1:
        recon_timeseries = np.reshape(recon_timeseries,
                                  [recon_timeseries.shape[0], 1])

    #####
    # convert the recon_timeseries to the standard
    # space brain dipole grid at the specfied resolution 
    recon_ts_out, reference_brain_fname = resample_recon_ts \
        (subjects_dir, subject,
         recon_timeseries=recon_timeseries,
         spatial_resolution=None,
         reference_brain=reference_brain)

    #####
    # output recon_ts_out as niftii file

    mni_nii_nib = nib.load(reference_brain_fname)
    coords_mni = rhino_utils.niimask2indexpointcloud(reference_brain_fname).T

    mni_nii_values = mni_nii_nib.get_fdata()
    mni_nii_values = np.zeros(np.append((mni_nii_values.shape), recon_timeseries.shape[1]))

    for ii in range(recon_ts_out.shape[0]):
        try:
            mni_nii_values[coords_mni[ii, 0], coords_mni[ii, 1],
            coords_mni[ii, 2], :] = recon_ts_out[ii, :]
        except IndexError:
            print('Index out of bounds ignored')
            print('Coords are {}'.format(coords_mni[ii, :]))

    # import pdb; pdb.set_trace()

    # SAVE AS NIFTI
    vol_nii = nib.Nifti1Image(mni_nii_values, mni_nii_nib.affine)

    vol_nii.header.set_xyzt_units(2)  # mm
    if times is not None:
        vol_nii.header['pixdim'][4] = times[1] - times[0]
        vol_nii.header['toffset'] = -0.5
        vol_nii.header.set_xyzt_units(2, 8)  # mm and secs

    nib.save(vol_nii, out_nii_fname)

    # os.system('fslcpgeom {} {}'.format(reference_brain_fname, out_nii_fname))

    return out_nii_fname, reference_brain_fname


#############################################################################

def fsleyes(image_list):
    cmd = 'fsleyes '
    for img in image_list:
        cmd += img
        cmd += ' '

    cmd += '&'
    print(cmd)
    os.system(cmd)


#############################################################################

def fsleyes_overlay(background_img, overlay_img):
    if type(background_img) is str:
        if background_img == 'mni':
            mni_resolution = int(nib.load(overlay_img).header.get_zooms()[0])
            background_img = op.join(os.environ['FSLDIR'],
                                     'data/standard/MNI152_T1_{}mm_brain.nii.gz'.format(mni_resolution))
        elif background_img[0:3] == 'mni':
            mni_resolution = int(background_img[3])
            background_img = op.join(os.environ['FSLDIR'],
                                     'data/standard/MNI152_T1_{}mm_brain.nii.gz'.format(mni_resolution))

    cmd = 'fsleyes {} --volume 0 {} --alpha 100.0 --cmap red-yellow \
--negativeCmap blue-lightblue --useNegativeCmap &'.format(
        background_img,
        overlay_img)

    print(cmd)
    os.system(cmd)