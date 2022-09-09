#!/usr/bin/env python

"""Forward modelling in RHINO.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>


from mne import (
    make_bem_model,
    make_bem_solution,
    make_forward_solution,
    write_forward_solution,
)
from mne.bem import ConductorModel, read_bem_solution
from mne.transforms import read_trans
from mne.io import read_info
from mne.io.constants import FIFF

from osl.source_recon.rhino import get_coreg_filenames, setup_volume_source_space
from osl.utils.logger import log_or_print


def forward_model(
    subjects_dir,
    subject,
    model="Single Layer",
    gridstep=8,
    mindist=4.0,
    exclude=0.0,
    eeg=False,
    meg=True,
    verbose=False,
    logger=None,
):
    """Compute forward model.

    Parameters
    ----------
    subjects_dir : string
        Directory to find RHINO subject dirs in.
    subject : string
        Subject name dir to find RHINO files in.
    model : string
        'Single Layer' or 'Triple Layer'
        'Single Layer' to use single layer (brain/cortex)
        'Triple Layer' to three layers (scalp, inner skull, brain/cortex)
    gridstep : int
        A grid will be constructed with the spacing given by ``gridstep`` in mm,
        generating a volume source space.
    mindist : float
        Exclude points closer than this distance (mm) to the bounding surface.
    exclude : float
        Exclude points closer than this distance (mm) from the center of mass
        of the bounding surface.
    eeg : bool
        Whether to compute forward model for eeg sensors
    meg : bool
        Whether to compute forward model for meg sensors
    logger : logging.getLogger
        Logger.
    """
    log_or_print("*** RUNNING OSL RHINO FORWARD MODEL ***", logger)

    # compute MNE bem solution
    if model == "Single Layer":
        conductivity = (0.3,)  # for single layer
    elif model == "Triple Layer":
        conductivity = (0.3, 0.006, 0.3)  # for three layers
    else:
        raise ValueError("{} is an invalid model choice".format(model))

    vol_src = setup_volume_source_space(
        subjects_dir,
        subject,
        gridstep=gridstep,
        mindist=mindist,
        exclude=exclude,
        logger=logger,
    )

    # The BEM solution requires a BEM model which describes the geometry of the
    # head the conductivities of the different tissues. See:
    # https://mne.tools/stable/auto_tutorials/forward/30_forward.html#sphx-glr-auto-tutorials-forward-30-forward-py
    #
    # Note that the BEM does not involve any use of transforms between spaces.
    # The BEM only depends on the head geometry and conductivities.
    # It is therefore independent from the MEG data and the head position.
    #
    # This will get the surfaces from: subjects_dir/subject/bem/inner_skull.surf
    # which is where rhino.setup_volume_source_space will have put it.

    model = make_bem_model(
        subjects_dir=subjects_dir,
        subject=subject,
        ico=None,
        conductivity=conductivity,
        verbose=verbose,
    )

    bem = make_bem_solution(model)

    fwd = make_fwd_solution(
        subjects_dir,
        subject,
        src=vol_src,
        ignore_ref=True,
        bem=bem,
        eeg=eeg,
        meg=meg,
        verbose=verbose,
    )

    filenames = get_coreg_filenames(subjects_dir, subject)
    write_forward_solution(filenames["forward_model_file"], fwd, overwrite=True)

    log_or_print("*** OSL RHINO FORWARD MODEL COMPLETE ***", logger)


def make_fwd_solution(
    subjects_dir,
    subject,
    src,
    bem,
    meg=True,
    eeg=True,
    mindist=0.0,
    ignore_ref=False,
    n_jobs=1,
    verbose=None,
):
    """Calculate a forward solution for a subject. This is a RHINO wrapper
    for mne.make_forward_solution.

    Parameters
    ----------
    See mne.make_forward_solution for the full set of parameters, with the
    exception of:
    subjects_dir : string
        Directory to find RHINO subject dirs in.
    subject : string
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
    """

    fif_file = get_coreg_filenames(subjects_dir, subject)["fif_file"]

    # Note, forward model is done in Head space:
    head_mri_trans_file = get_coreg_filenames(subjects_dir, subject)["head_mri_t_file"]

    # Src should be in MRI space. Let's just check that is the
    # case
    if src[0]["coord_frame"] != FIFF.FIFFV_COORD_MRI:
        raise RuntimeError("src is not in MRI coordinates")

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
    head_mri_t["trans"][0:3, -1] = head_mri_t["trans"][0:3, -1] / 1000

    if isinstance(bem, str):
        bem = read_bem_solution(bem)
    else:
        if not isinstance(bem, ConductorModel):
            raise TypeError("bem must be a string or ConductorModel")

        bem = bem.copy()

    for ii in range(len(bem["surfs"])):
        bem["surfs"][ii]["tris"] = bem["surfs"][ii]["tris"].astype(int)

    info = read_info(fif_file)

    # -------------------------------------------------------------------------
    # Main MNE call
    fwd = make_forward_solution(
        info,
        trans=head_mri_t,
        src=src,
        bem=bem,
        eeg=eeg,
        meg=meg,
        mindist=mindist,
        ignore_ref=ignore_ref,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # fwd should be in Head space. Let's just check that is the case:
    if fwd["src"][0]["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
        raise RuntimeError("fwd['src'][0] is not in HEAD coordinates")

    return fwd
