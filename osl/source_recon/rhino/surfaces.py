"""Calculation of surfaces in RHINO.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import os.path as op
from pathlib import Path
from shutil import copyfile
from datetime import datetime

import cv2
import numpy as np
import nibabel as nib
from scipy.ndimage import morphology
from sklearn.mixture import GaussianMixture

from mne.transforms import write_trans, Transform

import osl.source_recon.rhino.utils as rhino_utils
from osl.utils.logger import log_or_print


def get_surfaces_filenames(subjects_dir, subject):
    """Generates a dict of files generated and used by rhino.compute_surfaces.

    Files will be in subjects_dir/subject/rhino/surfaces.

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories.
    subject : string
        Subject directory name to put the surfaces in.

    Returns
    -------
    filenames : dict
        A dict of files generated and used by rhino.compute_surfaces.
        Note that  due to the unusal naming conventions used by BET:
        - bet_inskull_*_file is actually the brain surface
        - bet_outskull_*_file is actually the inner skull surface
        - bet_outskin_*_file is the outer skin/scalp surface
    """
    basedir = op.join(subjects_dir, subject, "rhino", "surfaces")
    os.makedirs(basedir, exist_ok=True)

    filenames = {
        "basedir": basedir,
        "smri_file": op.join(basedir, "smri.nii.gz"),
        "mni2mri_flirt_xform_file": op.join(basedir, "mni2mri_flirt_xform_file.txt"),
        "mni_mri_t_file": op.join(basedir, "mni_mri-trans.fif"),
        "bet_outskin_mesh_vtk_file": op.join(basedir, "outskin_mesh.vtk"),  # BET output
        "bet_inskull_mesh_vtk_file": op.join(basedir, "inskull_mesh.vtk"),  # BET output
        "bet_outskull_mesh_vtk_file": op.join(
            basedir, "outskull_mesh.vtk"
        ),  # BET output
        "bet_outskin_mesh_file": op.join(basedir, "outskin_mesh.nii.gz"),
        "bet_outskin_plus_nose_mesh_file": op.join(
            basedir, "outskin_plus_nose_mesh.nii.gz"
        ),
        "bet_inskull_mesh_file": op.join(basedir, "inskull_mesh.nii.gz"),
        "bet_outskull_mesh_file": op.join(basedir, "outskull_mesh.nii.gz"),
        "std_brain": op.join(
            os.environ["FSLDIR"],
            "data",
            "standard",
            "MNI152_T1_1mm_brain.nii.gz",
        ),
        "std_brain_bigfov": op.join(
            os.environ["FSLDIR"],
            "data",
            "standard",
            "MNI152_T1_1mm_BigFoV_facemask.nii.gz",
        ),
        "completed": op.join(basedir, "completed.txt"),
    }

    return filenames


def compute_surfaces(
    smri_file,
    subjects_dir,
    subject,
    include_nose=True,
    cleanup_files=True,
    recompute_surfaces=False,
):
    """Compute surfaces.

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
    6) Output the transform from sMRI space to MNI
    7) Output surfaces in sMRI space

    Parameters
    ----------
    smri_file : string
        Full path to structural MRI in niftii format
        (with .nii.gz extension).
        This is assumed to have a valid sform, i.e. the sform code
        needs to be 4 or 1, and the sform should transform from voxel
        indices to voxel coords in mm. The axis sform uses to do this
        will be the native/sMRI axis used throughout rhino. The qform
        will be ignored.
    subjects_dir : string
        Directory to put RHINO subject directories in.
        Files will be in subjects_dir/subject/surfaces.
    subject : string
        Subject name directory to put RHINO files in.
        Files will be in subjects_dir/subject/surfaces.
    include_nose : bool
        Specifies whether to add the nose to the outer skin
        (scalp) surface. This can help rhino's coreg to work
        better, assuming that there are headshape points that also
        include the nose.
        Requires the smri_file to have a FOV that includes the nose!
    cleanup_files : bool
        Specifies whether to cleanup intermediate files in the coreg
        directory.
    recompute_surfaces : bool
        Specifies whether or not to run compute_surfaces if the passed in
        options have already been run.
    """

    # Note the jargon used varies for xforms and coord spaces, e.g.:
    # MEG (device) -- dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus)-- head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native) -- mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices
    # MRI (native) -- sform (mri2mniaxes) --> MNI axes
    #
    # RHINO does everthing in mm

    filenames = get_surfaces_filenames(subjects_dir, subject)

    if not recompute_surfaces:
        # Check if surfaces have already been computed
        if Path(filenames["completed"]).exists():
            with open(filenames["completed"], "r") as file:
                lines = file.readlines()
                completed_mri_file = lines[1].split(":")[1].strip()
                completed_include_nose = lines[2].split(":")[1].strip() == "True"
                is_same_mri = completed_mri_file == filenames["smri_file"]
                is_same_include_nose = completed_include_nose == include_nose
                if is_same_mri and is_same_include_nose:
                    log_or_print("*** OSL RHINO: USING PREVIOUSLY COMPUTED SURFACES ***")
                    log_or_print(f"Surfaces directory: {filenames['basedir']}")
                    log_or_print(f"include_nose={completed_include_nose}")
                    return

    log_or_print("*** RUNNING OSL RHINO COMPUTE SURFACES ***")
    if include_nose:
        log_or_print("The nose is going to be added to the outer skin (scalp) surface.")
        log_or_print("Please ensure that the structural MRI has a FOV that includes the nose")
    else:
        log_or_print("The nose is not going to be added to the outer skin (scalp) surface")

    # Check smri_file
    smri_ext = "".join(Path(smri_file).suffixes)
    if smri_ext not in [".nii", ".nii.gz"]:
        raise ValueError(
            "smri_file needs to be a niftii file with a .nii or .nii.gz extension"
        )

    # Copy smri_name to new file for modification
    copyfile(smri_file, filenames["smri_file"])

    # RHINO will always use the sform, and so we will set the qform to be same
    # as sform for sMRI, to stop the original qform from being used by mistake
    # (e.g. by flirt)
    cmd = "fslorient -copysform2qform {}".format(filenames["smri_file"])
    rhino_utils.system_call(cmd)

    # We will assume orientation of standard brain is RADIOLOGICAL
    # But let's check that is the case:
    std_orient = rhino_utils._get_orient(filenames["std_brain"])
    if std_orient != "RADIOLOGICAL":
        raise ValueError(
            "Orientation of standard brain must be RADIOLOGICAL, \
please check output of:\n fslorient -orient {}".format(
                filenames["std_brain"]
            )
        )

    # We will assume orientation of sMRI brain is RADIOLOGICAL
    # But let's check that is the case:
    smri_orient = rhino_utils._get_orient(filenames["smri_file"])

    if smri_orient != "RADIOLOGICAL" and smri_orient != "NEUROLOGICAL":
        raise ValueError(
            "Cannot determine orientation of subject brain, \
please check output of:\n fslorient -orient {}".format(
                filenames["smri_file"]
            )
        )

    # if orientation is not RADIOLOGICAL then force it to be RADIOLOGICAL
    if smri_orient != "RADIOLOGICAL":
        log_or_print("reorienting subject brain to be RADIOLOGICAL")
        rhino_utils.system_call(
            "fslorient -forceradiological {}".format(filenames["smri_file"])
        )

    log_or_print("You can use the following call to check the passed in structural MRI is appropriate,")
    log_or_print("including checking that the L-R, S-I, A-P labels are sensible:")
    log_or_print("In Python:")
    log_or_print('fsleyes("{}", "{}")'.format(filenames["smri_file"], filenames["std_brain"]))
    log_or_print("From the cmd line:")
    log_or_print("fsleyes {} {}".format(filenames["smri_file"], filenames["std_brain"]))

    # -------------------------------------------------------------------------
    # 1) Transform sMRI to be aligned with the MNI axes so that BET works well

    img = nib.load(filenames["smri_file"])
    img_density = np.sum(img.get_data()) / np.prod(img.get_data().shape)

    # We will start by transforming sMRI
    # so that its voxel indices axes are aligned to MNI's
    # This helps BET work.
    # CALCULATE mri2mniaxes
    flirt_mri2mniaxes_xform = rhino_utils._get_flirt_xform_between_axes(
        filenames["smri_file"], filenames["std_brain"]
    )

    # Write xform to disk so flirt can use it
    flirt_mri2mniaxes_xform_file = op.join(
        filenames["basedir"], "flirt_mri2mniaxes_xform.txt"
    )
    np.savetxt(flirt_mri2mniaxes_xform_file, flirt_mri2mniaxes_xform)

    # Apply mri2mniaxes xform to smri to get smri_mniaxes, which means sMRIs
    # voxel indices axes are aligned to be the same as MNI's
    flirt_smri_mniaxes_file = op.join(filenames["basedir"], "flirt_smri_mniaxes.nii.gz")
    rhino_utils.system_call(
        "flirt -in {} -ref {} -applyxfm -init {} -out {}".format(
            filenames["smri_file"],
            filenames["std_brain"],
            flirt_mri2mniaxes_xform_file,
            flirt_smri_mniaxes_file,
        )
    )

    img = nib.load(flirt_smri_mniaxes_file)
    img_latest_density = np.sum(img.get_data()) / np.prod(img.get_data().shape)

    if 5 * img_latest_density < img_density:
        raise Exception(
            "Something is wrong with the passed in structural MRI:\n   {}\n"
            "Either it is empty or the sformcode is incorrectly set.\n"
            "Try running the following from a cmd line: \n"
            "   fsleyes {} {} \n"
            "And see if the standard space brain is shown in the same postcode as the structural.\n"
            "If it is not, then the sformcode in the structural image needs setting correctly.\n".format(
                filenames["smri_file"], filenames["std_brain"], filenames["smri_file"]
            )
        )

    # -------------------------------------------------------------------------
    # 2) Use BET to skull strip sMRI so that flirt works well
    log_or_print("Running BET pre-FLIRT...")

    flirt_smri_mniaxes_bet_file = op.join(
        filenames["basedir"], "flirt_smri_mniaxes_bet"
    )
    rhino_utils.system_call(
        "bet2 {} {}".format(flirt_smri_mniaxes_file, flirt_smri_mniaxes_bet_file)
    )

    # -------------------------------------------------------------------------
    # 3) Use flirt to register skull stripped sMRI to MNI space
    log_or_print("Running FLIRT...")

    # Flirt is run on the skull stripped brains to register the smri_mniaxes
    # to the MNI standard brain

    flirt_mniaxes2mni_file = op.join(filenames["basedir"], "flirt_mniaxes2mni.txt")
    flirt_smri_mni_bet_file = op.join(filenames["basedir"], "flirt_smri_mni_bet.nii.gz")
    rhino_utils.system_call(
        "flirt -in {} -ref {} -omat {} -o {}".format(
            flirt_smri_mniaxes_bet_file,
            filenames["std_brain"],
            flirt_mniaxes2mni_file,
            flirt_smri_mni_bet_file,
        )
    )

    # Calculate overall flirt transform, from mri to MNI
    mri2mni_flirt_xform_file = op.join(
        filenames["basedir"], "flirt_mri2mni_flirt_xform.txt"
    )
    rhino_utils.system_call(
        "convert_xfm -omat {} -concat {} {}".format(
            mri2mni_flirt_xform_file,
            flirt_mniaxes2mni_file,
            flirt_mri2mniaxes_xform_file,
        )
    )

    # and also calculate its inverse, from MNI to mri
    mni2mri_flirt_xform_file = filenames["mni2mri_flirt_xform_file"]

    rhino_utils.system_call(
        "convert_xfm -omat {}  -inverse {}".format(
            mni2mri_flirt_xform_file, mri2mni_flirt_xform_file
        )
    )

    # move full smri into MNI space to do full bet and betsurf
    flirt_smri_mni_file = op.join(filenames["basedir"], "flirt_smri_mni.nii.gz")
    rhino_utils.system_call(
        "flirt -in {} -ref {} -applyxfm -init {} -out {}".format(
            filenames["smri_file"],
            filenames["std_brain"],
            mri2mni_flirt_xform_file,
            flirt_smri_mni_file,
        )
    )

    # -------------------------------------------------------------------------
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

    # Run BET on smri to get the surface mesh (in MNI space),
    # as BETSURF needs this.
    log_or_print("Running BET pre-BETSURF...")

    flirt_smri_mni_bet_file = op.join(filenames["basedir"], "flirt_smri_mni_bet")
    rhino_utils.system_call(
        "bet2 {} {} --mesh".format(flirt_smri_mni_file, flirt_smri_mni_bet_file)
    )

    # Run BETSURF - to get the head surfaces in MNI space
    log_or_print("Running BETSURF...")

    # Need to provide BETSURF with transform to MNI space.
    # Since flirt_smri_mni_file is already in MNI space,
    # this will just be the identity matrix

    flirt_identity_xform_file = op.join(
        filenames["basedir"], "flirt_identity_xform.txt"
    )
    np.savetxt(flirt_identity_xform_file, np.eye(4))

    bet_mesh_file = op.join(flirt_smri_mni_bet_file + "_mesh.vtk")
    rhino_utils.system_call(
        "betsurf --t1only -o {} {} {} {}".format(
            flirt_smri_mni_file,
            bet_mesh_file,
            flirt_identity_xform_file,
            op.join(filenames["basedir"], "flirt"),
        )
    )

    # -------------------------------------------------------------------------
    # 5) Refine scalp outline, adding nose to scalp surface (optional)
    log_or_print("Refining scalp surface...")

    # We do this in MNI big FOV space, to allow the full nose to be included

    # Calculate flirt_mni2mnibigfov_xform
    mni2mnibigfov_xform = rhino_utils._get_flirt_xform_between_axes(
        from_nii=flirt_smri_mni_file, target_nii=filenames["std_brain_bigfov"]
    )

    flirt_mni2mnibigfov_xform_file = op.join(
        filenames["basedir"], "flirt_mni2mnibigfov_xform.txt"
    )
    np.savetxt(flirt_mni2mnibigfov_xform_file, mni2mnibigfov_xform)

    # Calculate overall transform, from smri to MNI big fov
    flirt_mri2mnibigfov_xform_file = op.join(
        filenames["basedir"], "flirt_mri2mnibigfov_xform.txt"
    )
    rhino_utils.system_call(
        "convert_xfm -omat {} -concat {} {}".format(
            flirt_mri2mnibigfov_xform_file,
            flirt_mni2mnibigfov_xform_file,
            mri2mni_flirt_xform_file,
        )
    )

    # move MRI to MNI big FOV space and load in
    flirt_smri_mni_bigfov_file = op.join(filenames["basedir"], "flirt_smri_mni_bigfov")
    rhino_utils.system_call(
        "flirt -in {} -ref {} -applyxfm -init {} -out {}".format(
            filenames["smri_file"],
            filenames["std_brain_bigfov"],
            flirt_mri2mnibigfov_xform_file,
            flirt_smri_mni_bigfov_file,
        )
    )
    vol = nib.load(flirt_smri_mni_bigfov_file + ".nii.gz")

    # move scalp to MNI big FOV space and load in
    flirt_outskin_file = op.join(filenames["basedir"], "flirt_outskin_mesh")
    flirt_outskin_bigfov_file = op.join(
        filenames["basedir"], "flirt_outskin_mesh_bigfov"
    )
    rhino_utils.system_call(
        "flirt -in {} -ref {} -applyxfm -init {} -out {}".format(
            flirt_outskin_file,
            filenames["std_brain_bigfov"],
            flirt_mni2mnibigfov_xform_file,
            flirt_outskin_bigfov_file,
        )
    )
    scalp = nib.load(flirt_outskin_bigfov_file + ".nii.gz")

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

    if include_nose:
        log_or_print("Adding nose to scalp surface...")

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
        gm = GaussianMixture(n_components=2, random_state=0, covariance_type="diag")
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

    # EXTRACT OUTLINE
    outline = np.zeros(mask.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = mask.astype(np.uint8)

    # Use morph gradient to find the outline of the solid mask
    for i in range(outline.shape[0]):
        outline[i, :, :] += cv2.morphologyEx(mask[i, :, :], cv2.MORPH_GRADIENT, kernel)
    for i in range(outline.shape[1]):
        outline[:, i, :] += cv2.morphologyEx(mask[:, i, :], cv2.MORPH_GRADIENT, kernel)
    for i in range(50, 300, 1):
        outline[:, :, i] += cv2.morphologyEx(mask[:, :, i], cv2.MORPH_GRADIENT, kernel)
    outline /= 3

    outline[np.where(outline > 0.6)] = 1
    outline[np.where(outline <= 0.6)] = 0
    outline = outline.astype(np.uint8)

    # SAVE AS NIFTI
    outline_nii = nib.Nifti1Image(outline, scalp.affine)

    nib.save(outline_nii, op.join(flirt_outskin_bigfov_file + "_plus_nose.nii.gz"))

    rhino_utils.system_call(
        "fslcpgeom {} {}".format(
            op.join(flirt_outskin_bigfov_file + ".nii.gz"),
            op.join(flirt_outskin_bigfov_file + "_plus_nose.nii.gz"),
        )
    )

    # Transform outskin plus nose nii mesh from MNI big FOV to MRI space

    # first we need to invert the flirt_mri2mnibigfov_xform_file xform:
    flirt_mnibigfov2mri_xform_file = op.join(
        filenames["basedir"], "flirt_mnibigfov2mri_xform.txt"
    )
    rhino_utils.system_call(
        "convert_xfm -omat {} -inverse {}".format(
            flirt_mnibigfov2mri_xform_file, flirt_mri2mnibigfov_xform_file
        )
    )

    rhino_utils.system_call(
        "flirt -in {} -ref {} -applyxfm -init {} -out {}".format(
            op.join(flirt_outskin_bigfov_file + "_plus_nose.nii.gz"),
            filenames["smri_file"],
            flirt_mnibigfov2mri_xform_file,
            filenames["bet_outskin_plus_nose_mesh_file"],
        )
    )

    # -------------------------------------------------------------------------
    # 6) Output the transform from sMRI space to MNI

    flirt_mni2mri = np.loadtxt(mni2mri_flirt_xform_file)

    xform_mni2mri = rhino_utils._get_mne_xform_from_flirt_xform(
        flirt_mni2mri, filenames["std_brain"], filenames["smri_file"]
    )

    mni_mri_t = Transform("mni_tal", "mri", xform_mni2mri)
    write_trans(filenames["mni_mri_t_file"], mni_mri_t, overwrite=True)

    # -------------------------------------------------------------------------
    # 7) Output surfaces in sMRI(native) space

    # Transform betsurf output mask/mesh output from MNI to sMRI space
    rhino_utils._transform_bet_surfaces(
        mni2mri_flirt_xform_file,
        filenames["mni_mri_t_file"],
        filenames,
        filenames["smri_file"],
    )

    # -------------------------------------------------------------------------
    # Write a file to indicate RHINO has been run

    with open(filenames["completed"], "w") as file:
        file.write(f"Completed: {datetime.now()}\n")
        file.write(f"MRI file: {filenames['smri_file']}\n")
        file.write(f"Include nose: {include_nose}\n")

    # -------------------------------------------------------------------------
    # Clean up

    if cleanup_files:
        # CLEAN UP FILES ON DISK
        rhino_utils.system_call(
            "rm -f {}".format(op.join(filenames["basedir"], "flirt*"))
        )

    log_or_print(
        'rhino.surfaces_display("{}", "{}") can be used to check the result'.format(
            subjects_dir, subject
        )
    )
    log_or_print("*** OSL RHINO COMPUTE SURFACES COMPLETE ***")


def surfaces_display(subjects_dir, subject):
    """Display surfaces.

    Displays the surfaces extracted from the sMRI using
    rhino.compute_surfaces.

    Display is shown in sMRI (native) space.

    Parameters
    ----------
    subjects_dir : string
        Directory to put RHINO subject directories in.
        Files will be in subjects_dir/subject/surfaces.
    subject : string
        Subject name directory to put RHINO files in.
        Files will be in subjects_dir/subject/surfaces.

    Notes
    -----
    bet_inskull_mesh_file is actually the brain surface and
    bet_outskull_mesh_file is the inner skull surface, due to
    the naming conventions used by BET.
    """

    filenames = get_surfaces_filenames(subjects_dir, subject)

    rhino_utils.system_call(
        "fsleyes {} {} {} {} {} &".format(
            filenames["smri_file"],
            filenames["bet_inskull_mesh_file"],
            filenames["bet_outskin_mesh_file"],
            filenames["bet_outskull_mesh_file"],
            filenames["bet_outskin_plus_nose_mesh_file"],
        )
    )
