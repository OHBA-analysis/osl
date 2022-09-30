#!/usr/bin/env python

"""Wrappers for fsleyes.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import os.path as op

import nibabel as nib

import osl.source_recon.rhino.utils as rhino_utils


def setup_fsl(directory):
    """Setup FSL.

    Parameters
    ----------
    directory : str
        Path to FSL installation.
    """
    if "FSLDIR" not in os.environ:
        os.environ["FSLDIR"] = directory
    if "{:s}/bin" not in os.getenv("PATH"):
        os.environ["PATH"] = "{:s}/bin:{:s}".format(directory, os.getenv("PATH"))
    if "FSLOUTPUTTYPE" not in os.environ:
        os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"


def check_fsl():
    """Check FSL is installed."""
    if "FSLDIR" not in os.environ:
        raise RuntimeError("Please setup FSL, e.g. with osl.source_recon.setup_fsl().")


def fsleyes(image_list):
    """Displays list of niftii's using external command line call to fsleyes.

    Parameters
    ----------
    image_list : string or tuple of strings
        Niftii filenames or tuple of niftii filenames

    Examples
    --------
    fsleyes(image)
    fsleyes((image1, image2))
    """

    # Check if image_list is a single file name
    if isinstance(image_list, str):
        image_list = (image_list,)

    cmd = "fsleyes "
    for img in image_list:
        cmd += img
        cmd += " "
    cmd += "&"

    rhino_utils.system_call(cmd, verbose=True)


def fsleyes_overlay(background_img, overlay_img):
    """Displays overlay_img and background_img using external command line call
    to fsleyes.

    Parameters
    ----------
    background_img : string
        Background niftii filename
    overlay_img : string
        Overlay niftii filename
    """

    if type(background_img) is str:
        if background_img == "mni":
            mni_resolution = int(nib.load(overlay_img).header.get_zooms()[0])
            background_img = op.join(
                os.environ["FSLDIR"],
                "data/standard/MNI152_T1_{}mm_brain.nii.gz".format(mni_resolution),
            )
        elif background_img[0:3] == "mni":
            mni_resolution = int(background_img[3])
            background_img = op.join(
                os.environ["FSLDIR"],
                "data/standard/MNI152_T1_{}mm_brain.nii.gz".format(mni_resolution),
            )

    cmd = (
        "fsleyes {} --volume 0 {} --alpha 100.0 ".format(background_img, overlay_img)
        + "--cmap red-yellow --negativeCmap blue-lightblue --useNegativeCmap &"
    )

    rhino_utils.system_call(cmd)
