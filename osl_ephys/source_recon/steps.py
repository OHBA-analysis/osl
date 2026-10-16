"""The coreg, source_recon, parcellation and group steps of a config.

Each step takes the dataset dict of a session and its options, builds the
:class:`~osl_ephys.source_recon.session.Session` that holds the session's
files, and calls the function that does the work. The v2 step names are
kept; v2 options that no longer apply are ignored with a warning.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import mne
import numpy as np

from osl_ephys import files
from osl_ephys.pipeline.registry import step
from osl_ephys.source_recon import beamforming, parcellation, rhino, sign_flipping
from osl_ephys.source_recon.session import Session

logger = logging.getLogger("osl_ephys")


def session_files(dataset: dict) -> Session:
    """The Session of a dataset, built from its manifest row on first use.

    A session with no ``smri_file`` uses the MNI152 standard brain surfaces
    from osl-files in place of its own.
    """
    if "session_files" not in dataset:
        row, paths = dataset["session"], dataset["paths"]
        surfaces_dir = (
            paths["surfaces_dir"]
            if row.get("smri_file")
            else files.mni152_surfaces.directory
        )
        preproc_file = paths.get("preproc_file")
        dataset["session_files"] = Session(
            outdir=str(paths["session_dir"].parent),
            id=row["id"],
            surfaces_dir=str(surfaces_dir),
            preproc_file=str(preproc_file) if preproc_file is not None else None,
            pos_file=row.get("pos_file"),
            elc_file=row.get("elc_file"),
            head_model_id=row.get("head_model_id"),
        )
    return dataset["session_files"]


def _ignored(userargs: dict, *names: str) -> None:
    """Drop v2 options that no longer apply, saying so."""
    for name in names:
        if name in userargs:
            warnings.warn(f"'{name}' is no longer used and was ignored.", FutureWarning)
            userargs.pop(name)


def _renamed(userargs: dict, old: str, new: str) -> None:
    if old in userargs:
        userargs[new] = userargs.pop(old)


@step("coreg", per="subject")
@step("coreg", name="compute_surfaces", per="subject")
def extract_surfaces(dataset, userargs):
    """Extract the head surfaces from the subject's structural MRI with FSL.

    Options: ``include_nose``, ``do_mri2mniaxes_xform``, ``bet_fval``,
    ``nonlinear_registration``. Skipped for a session with no ``smri_file``,
    which uses the MNI152 surfaces.
    """
    _ignored(userargs, "cleanup_files", "recompute_surfaces", "use_qform")
    row = dataset["session"]
    if not row.get("smri_file"):
        logger.info(f"{row['id']} has no structural MRI; the MNI152 surfaces are used.")
        return dataset
    rhino.extract_surfaces(
        str(row["smri_file"]), str(dataset["paths"]["surfaces_dir"]), **userargs
    )
    return dataset


@step("coreg")
@step("coreg", name="extract_polhemus_from_info")
@step("coreg", name="extract_fiducials_from_fif")
def extract_fiducials_and_headshape(dataset, userargs):
    """Fiducials and headshape points from the preprocessed file's digitisation.

    Options: ``include_eeg_as_headshape``, ``include_hpi_as_headshape``,
    ``include_extra_as_headshape``.
    """
    _ignored(userargs, "rescale")
    rhino.extract_fiducials_and_headshape_from_fif(session_files(dataset), **userargs)
    return dataset


@step("coreg")
def extract_polhemus_from_pos(dataset, userargs):
    """Fiducials and headshape points from the session's ``pos_file``."""
    rhino.extract_fiducials_and_headshape_from_pos(session_files(dataset))
    return dataset


@step("coreg")
def extract_polhemus_from_elc(dataset, userargs):
    """Fiducials and headshape points from the session's ``elc_file``. Option: ``remove_nose``."""
    rhino.extract_fiducials_and_headshape_from_elc(session_files(dataset), **userargs)
    return dataset


@step("coreg")
def remove_stray_headshape_points(dataset, userargs):
    """Remove headshape points on the nose, neck or far from the head. Options: ``nose``, ``neck``, ``far``."""
    rhino.remove_stray_headshape_points(session_files(dataset), **userargs)
    return dataset


@step("coreg")
def save_mni_fiducials(dataset, userargs):
    """Use fiducials given in MNI coordinates for the coregistration.

    Option: ``filepath``, a text file with the nasion, LPA and RPA as rows
    (mm, MNI space). The coregistration step maps them to the subject.
    """
    coords = np.loadtxt(userargs["filepath"])
    dataset["mni_fiducials"] = {"nasion": coords[0], "lpa": coords[1], "rpa": coords[2]}
    return dataset


@step("coreg")
def coregister(dataset, userargs):
    """Coregister the head (sensors and headshape) with the MRI.

    Options: ``use_headshape``, ``use_nose``, ``allow_mri_scaling`` (v2
    ``allow_smri_scaling``), ``n_init``, ``plot_type``,
    ``already_coregistered`` (just save the files RHINO needs).
    """
    _renamed(userargs, "allow_smri_scaling", "allow_mri_scaling")
    _ignored(userargs, "use_dev_ctf_t", "surface_extraction_method")
    session = session_files(dataset)
    if userargs.pop("already_coregistered", False):
        rhino.save_coregistration_files(session)
        return dataset
    if userargs.get("n_init") is None:
        userargs["n_init"] = 1
    rhino.coregister_head_and_mri(
        session, mni_fiducials=dataset.get("mni_fiducials"), **userargs
    )
    return dataset


@step("source_recon")
def forward_model(dataset, userargs):
    """Forward model on a volumetric grid. Options: ``model``, ``gridstep``, ``mindist``, ``exclude``, ``eeg``, ``meg``."""
    _ignored(userargs, "surface_extraction_method")
    if userargs.pop("source_space", "volumetric") != "volumetric":
        raise NotImplementedError(
            "Surface source spaces go through the minimum_norm step."
        )
    rhino.forward_model(session_files(dataset), **userargs)
    return dataset


@step("source_recon")
def beamform(dataset, userargs):
    """LCMV beamformer filters, saved with the session.

    Options: ``chantypes``, ``rank``, ``pick_ori``, ``reduce_rank``,
    ``frequency_range`` (v2 ``freq_range``), ``use_bilateral_pairs``,
    ``bilateral_tol``, ``bilateral_tol_midline``.
    """
    _renamed(userargs, "freq_range", "frequency_range")
    _ignored(userargs, "weight_norm", "reg", "dipole_patches_file", "group_patches")
    beamforming.lcmv_beamformer(session_files(dataset), **userargs)
    return dataset


@step("parcellation")
def parcellate(dataset, userargs):
    """Parcel time courses from the LCMV filters, saved as ``<id>_parc-raw.fif``.

    Options: ``parcellation_file``, ``orthogonalisation``, ``extra_chans``
    (channels copied from the preprocessed data, ``stim`` by default),
    ``qc`` (save the QC plots, on by default).
    """
    _ignored(
        userargs,
        "method",
        "source_space",
        "surface_extraction_method",
        "source_method",
        "spatial_resolution",
        "reference_brain",
        "voxel_trans",
        "neighbour_distance",
    )
    session = session_files(dataset)
    extra_chans = userargs.pop("extra_chans", "stim")
    qc = userargs.pop("qc", True)
    parcel_data = parcellation.parcellate_lcmv(session, **userargs)
    raw = mne.io.read_raw_fif(session.preproc_file, preload=True, verbose=False)
    parc_file = dataset["paths"]["parc_file"]
    parc_file.parent.mkdir(parents=True, exist_ok=True)
    parcellation.save_as_fif(parcel_data, raw, str(parc_file), extra_chans=extra_chans)
    if qc and dataset.get("qc_dir") is not None:
        parcellation.save_qc_plots(
            dataset["qc_dir"],
            session.id,
            str(parc_file),
            userargs["parcellation_file"],
        )
    return dataset


@step("parcellation")
def beamform_and_parcellate(dataset, userargs):
    """``beamform`` then ``parcellate`` with the options of both."""
    beam_args = {
        k: userargs.pop(k)
        for k in list(userargs)
        if k
        in (
            "chantypes",
            "rank",
            "pick_ori",
            "reduce_rank",
            "frequency_range",
            "freq_range",
            "use_bilateral_pairs",
            "bilateral_tol",
            "bilateral_tol_midline",
            "weight_norm",
            "reg",
        )
    }
    dataset = beamform(dataset, beam_args)
    return parcellate(dataset, userargs)


def _parc_file(dataset: dict, id_: str) -> Path:
    """The parcellated file of a session, in the same output directory."""
    outdir = dataset["paths"]["session_dir"].parent
    return outdir / id_ / f"{id_}_parc-raw.fif"


def _template_cov(parc_file, n_embeddings, standardize):
    raw = mne.io.read_raw_fif(str(parc_file), preload=True, verbose=False)
    return sign_flipping.calc_cov(
        raw.get_data(picks="misc").T, n_embeddings, standardize
    )


@step("parcellation")
def fix_sign_ambiguity(dataset, userargs):
    """Flip the parcel signs of this session to match a template session.

    Options: ``template`` (the id of the template session), ``n_embeddings``,
    ``standardize``. Writes ``<id>_sflip_parc-raw.fif``.
    """
    _ignored(userargs, "n_init", "n_iter", "max_flips", "method", "refine", "n_refine")
    n_embeddings = userargs.get("n_embeddings", 15)
    standardize = userargs.get("standardize", True)
    cov = _template_cov(
        _parc_file(dataset, userargs["template"]), n_embeddings, standardize
    )
    id_ = dataset["session"]["id"]
    out = _parc_file(dataset, id_).with_name(f"{id_}_sflip_parc-raw.fif")
    sign_flipping.sign_flip_mne_raw(
        str(_parc_file(dataset, id_)), cov, str(out), n_embeddings, standardize
    )
    return dataset


@step("group")
def sign_flip(dataset, userargs):
    """Pick the template session and flip every session's parcel signs to it.

    The template is the session whose embedded covariance correlates best
    with the others' on average (or ``template`` if given). Options:
    ``n_embeddings``, ``standardize``, ``template``.
    """
    userargs.pop("run_on_group", None)
    n_embeddings = userargs.get("n_embeddings", 15)
    standardize = userargs.get("standardize", True)
    sessions = dataset["sessions"]
    outdir = Path(dataset["outdir"])
    parc_files = {
        s["id"]: outdir / s["id"] / f"{s['id']}_parc-raw.fif" for s in sessions
    }
    covs = {
        id_: _template_cov(f, n_embeddings, standardize)
        for id_, f in parc_files.items()
        if f.exists()
    }
    template = userargs.get("template")
    if template is None:
        scores = {
            a: np.mean(
                [
                    sign_flipping.calc_corr(covs[a], covs[b], n_embeddings)
                    for b in covs
                    if b != a
                ]
                or [0]
            )
            for a in covs
        }
        template = max(scores, key=scores.get)
        logger.info(f"Template session: {template}")
    dataset["template"] = template
    for id_, f in parc_files.items():
        if id_ in covs:
            out = f.with_name(f"{id_}_sflip_parc-raw.fif")
            sign_flipping.sign_flip_mne_raw(
                str(f), covs[template], str(out), n_embeddings, standardize
            )
    return dataset
