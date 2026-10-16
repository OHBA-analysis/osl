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
from osl_ephys.source_recon import (
    beamforming,
    freesurfer,
    parcellation,
    rhino,
    sign_flipping,
)
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
        # The FreeSurfer subject is the manifest's subject (recon-all is per subject)
        dataset["session_files"].subject = row.get("subject", row["id"])
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


def _freesurfer_route(dataset: dict, userargs: dict) -> bool:
    """Whether a step takes the FreeSurfer route: the step's own
    ``surface_extraction_method`` option, else the run's."""
    method = userargs.pop("surface_extraction_method", None) or dataset.get(
        "surface_extraction_method", "fsl"
    )
    return method == "freesurfer"


@step("coreg", per="subject")
def recon_all(dataset, userargs):
    """FreeSurfer's ``recon-all`` on the subject's structural MRI (needs FreeSurfer).

    A session with no ``smri_file`` gets MNE's fsaverage instead.
    """
    row = dataset["session"]
    if not row.get("smri_file"):
        logger.info(f"{row['id']} has no structural MRI; fsaverage is used.")
        freesurfer.fetch_fsaverage(session_files(dataset))
        return dataset
    freesurfer.recon_all(session_files(dataset), str(row["smri_file"]))
    return dataset


@step("coreg", per="subject")
def make_watershed_bem(dataset, userargs):
    """FreeSurfer's watershed BEM surfaces for the subject (needs FreeSurfer)."""
    _ignored(userargs, "overwrite")
    freesurfer.make_watershed_bem(session_files(dataset), **userargs)
    return dataset


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
def rescale_sensor_positions(dataset, userargs):
    """Rescale the sensor positions of the preprocessed file (EEG montages
    that do not fit the head). Option: ``rescale``, the ``[x, y, z]`` scale
    factors.
    """
    xscale, yscale, zscale = userargs.get("rescale", [1, 1, 1])
    rhino.rescale_sensor_positions(
        session_files(dataset).preproc_file, xscale, yscale, zscale
    )
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

    RHINO route options: ``use_headshape``, ``use_nose``,
    ``allow_mri_scaling`` (v2 ``allow_smri_scaling``), ``n_init``,
    ``plot_type``, ``already_coregistered`` (just save the files RHINO
    needs). FreeSurfer route options: ``allow_mri_scaling``, ``n_init``,
    ``fit_fiducials``, ``fit_icp``.
    """
    _renamed(userargs, "allow_smri_scaling", "allow_mri_scaling")
    _ignored(userargs, "use_dev_ctf_t")
    session = session_files(dataset)
    if _freesurfer_route(dataset, userargs):
        for name in ("use_headshape", "use_nose", "plot_type", "already_coregistered"):
            userargs.pop(name, None)
        if userargs.get("n_init") is None:
            userargs["n_init"] = 20
        freesurfer.coregister(session, **userargs)
        return dataset
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
    """Forward model.

    RHINO route (volumetric grid): ``model``, ``gridstep``, ``mindist``,
    ``exclude``, ``eeg``, ``meg``. FreeSurfer route, or
    ``source_space: surface``: ``source_space``, ``spacing`` (v2
    ``gridstep``), ``model``, ``conductivity``, ``ico``, ``mindist``,
    ``eeg``.
    """
    if _freesurfer_route(dataset, userargs) or userargs.get("source_space") in (
        "surface",
        "surf",
    ):
        _renamed(userargs, "gridstep", "spacing")
        userargs.pop("meg", None)
        userargs.pop("exclude", None)
        freesurfer.forward_model(session_files(dataset), **userargs)
        return dataset
    userargs.pop("source_space", None)
    rhino.forward_model(session_files(dataset), **userargs)
    return dataset


@step("source_recon")
def minimum_norm(dataset, userargs):
    """Minimum-norm inverse operator from the forward model.

    Options: ``chantypes``, ``rank``, ``depth``, ``loose``,
    ``frequency_range`` (v2 ``freq_range``). The RHINO route uses the
    volumetric forward model, the FreeSurfer route the surface one.
    ``lambda2`` and ``pick_ori`` belong to the parcellation step.
    """
    _renamed(userargs, "freq_range", "frequency_range")
    _ignored(userargs, "lambda2", "pick_ori")
    session = session_files(dataset)
    fwd_file = None if _freesurfer_route(dataset, userargs) else session.fwd_model_file
    freesurfer.minimum_norm(session, fwd_file=fwd_file, **userargs)
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
    """Parcel time courses from the source estimate, saved as ``<id>_parc-raw.fif``.

    Options: ``parcellation_file``, ``orthogonalisation`` (``symmetric`` or
    none), ``extra_chans`` (channels copied from the preprocessed data,
    ``stim`` by default), ``qc`` (save the QC plots, on by default), and
    ``source_method``: ``lcmv`` (default) uses the beamformer filters from
    the ``beamform`` step; ``minimum_norm`` applies the inverse operator
    from the ``minimum_norm`` step, volumetric on the RHINO route or on the
    surface on the FreeSurfer route, where ``parcellation_file`` is a
    FreeSurfer annotation (e.g. ``aparc``), ``method`` combines a label's
    vertices and ``morph`` is the subject whose annotation is used
    (fsaverage). Minimum-norm options: ``inverse_method``, ``lambda2``,
    ``pick_ori``.
    """
    _renamed(userargs, "freq_range", "frequency_range")
    _ignored(
        userargs,
        "source_space",
        "spatial_resolution",
        "voxel_trans",
        "neighbour_distance",
        "reference_brain",
    )
    session = session_files(dataset)
    source_method = str(userargs.pop("source_method", "lcmv")).lower()
    extra_chans = userargs.pop("extra_chans", "stim")
    qc = userargs.pop("qc", True)
    orthogonalisation = userargs.pop("orthogonalisation", None)
    parcellation_file = userargs.pop("parcellation_file")

    if source_method == "lcmv":
        for name in ("method", "morph", "inverse_method", "lambda2", "pick_ori"):
            userargs.pop(name, None)
        parcel_data = parcellation.parcellate_lcmv(
            session, parcellation_file, orthogonalisation=orthogonalisation, **userargs
        )
        names = None
        data = mne.io.read_raw_fif(session.preproc_file, preload=True, verbose=False)
    elif source_method in ("minimum_norm", "mne"):
        if _freesurfer_route(dataset, userargs):
            parcel_data, names, data = freesurfer.parcellate(
                session, parcellation_file, **userargs
            )
        else:
            userargs.pop("method", None)
            userargs.pop("morph", None)
            ts = freesurfer.apply_inverse_vol(
                session,
                method=userargs.pop("inverse_method", "MNE"),
                lambda2=userargs.pop("lambda2", 1.0 / 9),
            )
            # The dipoles are those of the RHINO forward model, on the MNI grid
            fwd = mne.read_forward_solution(session.fwd_model_file, verbose=False)
            coords = beamforming._get_source_coords_mni(session, fwd)
            parcel_data = parcellation.parcellate(
                ts, coords, parcellation_file, method="pca", orthogonalisation=None
            )
            names = None
            data = mne.io.read_raw_fif(
                session.preproc_file, preload=True, verbose=False
            )
        if orthogonalisation == "symmetric":
            parcel_data = parcellation._symmetric_orthogonalisation(parcel_data)
        elif orthogonalisation not in (None, "none", "None"):
            raise NotImplementedError(f"orthogonalisation '{orthogonalisation}'")
    else:
        raise ValueError("source_method must be 'lcmv' or 'minimum_norm'.")

    parc_file = dataset["paths"]["parc_file"]
    parc_file.parent.mkdir(parents=True, exist_ok=True)
    parc_raw = parcellation.convert_to_mne_raw(
        parcel_data, data, ch_names=names, extra_chans=extra_chans
    )
    parc_raw.save(str(parc_file), overwrite=True, verbose=False)
    if qc and dataset.get("qc_dir") is not None and source_method == "lcmv":
        parcellation.save_qc_plots(
            dataset["qc_dir"], session.id, str(parc_file), parcellation_file
        )
    return dataset


_BEAMFORM_OPTIONS = (
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
_MINIMUM_NORM_OPTIONS = (
    "chantypes",
    "rank",
    "depth",
    "loose",
    "frequency_range",
    "freq_range",
)


@step("parcellation")
def beamform_and_parcellate(dataset, userargs):
    """v2 name: ``beamform`` then ``parcellate`` with the options of both."""
    beam_args = {k: userargs.pop(k) for k in list(userargs) if k in _BEAMFORM_OPTIONS}
    dataset = beamform(dataset, beam_args)
    return parcellate(dataset, {**userargs, "source_method": "lcmv"})


@step("parcellation")
def minimum_norm_and_parcellate(dataset, userargs):
    """v2 name: ``minimum_norm`` then ``parcellate`` with ``source_method: minimum_norm``."""
    _renamed(userargs, "source_method", "inverse_method")
    mn_args = {k: userargs.pop(k) for k in list(userargs) if k in _MINIMUM_NORM_OPTIONS}
    if mn_args:
        dataset = minimum_norm(dataset, mn_args)
    return parcellate(dataset, {**userargs, "source_method": "minimum_norm"})


def _parc_file(dataset: dict, id_: str) -> Path:
    """The parcellated file of a session, in the same output directory."""
    outdir = dataset["paths"]["session_dir"].parent
    return outdir / id_ / f"{id_}_parc-raw.fif"


def _template_cov(parc_file, n_embeddings, standardize):
    raw = mne.io.read_raw_fif(str(parc_file), preload=True, verbose=False)
    return sign_flipping.calc_cov(
        raw.get_data(picks="misc").T, n_embeddings, standardize
    )


_SOLVERS = {
    "spectral": sign_flipping.find_flips_spectral,
    "belief_propagation": sign_flipping.find_flips_belief_propagation,
    "sdp": sign_flipping.find_flips_sdp,
}


def _flip_file(
    parc_file, template_cov, out, n_embeddings, standardize, method, options
):
    """Flip the parcel signs of one file to match the template covariance.

    ``method`` is ``coordinate_ascent`` (the default search) or one of the
    solvers in ``_SOLVERS``, with that solver's options.
    """
    if method in (None, "coordinate_ascent", "random"):
        sign_flipping.sign_flip_mne_raw(
            str(parc_file), template_cov, str(out), n_embeddings, standardize
        )
        return
    if method not in _SOLVERS:
        raise ValueError(
            f"method must be 'coordinate_ascent' or one of {sorted(_SOLVERS)}."
        )
    raw = mne.io.read_raw_fif(str(parc_file), preload=True, verbose=False)
    picks = mne.pick_types(raw.info, misc=True)
    cov = sign_flipping.calc_cov(raw.get_data(picks).T, n_embeddings, standardize)
    flips, metrics = _SOLVERS[method](cov, template_cov, n_embeddings, **options)
    logger.info(f"Template correlation {metrics[0]:.3f} -> {metrics[1]:.3f} ({method})")
    raw._data[picks] *= np.asarray(flips)[:, None]
    raw.save(str(out), overwrite=True, verbose=False)


@step("parcellation")
def fix_sign_ambiguity(dataset, userargs):
    """Flip the parcel signs of this session to match a template session.

    Options: ``template`` (the id of the template session), ``n_embeddings``,
    ``standardize``, ``method`` (``coordinate_ascent``, ``spectral``,
    ``belief_propagation`` or ``sdp``, plus that solver's options). Writes
    ``<id>_sflip_parc-raw.fif``.
    """
    _ignored(userargs, "n_init", "n_iter", "max_flips")
    n_embeddings = userargs.pop("n_embeddings", 15)
    standardize = userargs.pop("standardize", True)
    method = userargs.pop("method", "coordinate_ascent")
    template = userargs.pop("template")
    cov = _template_cov(_parc_file(dataset, template), n_embeddings, standardize)
    id_ = dataset["session"]["id"]
    out = _parc_file(dataset, id_).with_name(f"{id_}_sflip_parc-raw.fif")
    _flip_file(
        _parc_file(dataset, id_), cov, out, n_embeddings, standardize, method, userargs
    )
    return dataset


@step("group")
def sign_flip(dataset, userargs):
    """Pick the template session and flip every session's parcel signs to it.

    The template is the session whose embedded covariance correlates best
    with the others' on average (or ``template`` if given). Options:
    ``n_embeddings``, ``standardize``, ``template``, ``method`` (as for
    ``fix_sign_ambiguity``).
    """
    userargs.pop("run_on_group", None)
    n_embeddings = userargs.pop("n_embeddings", 15)
    standardize = userargs.pop("standardize", True)
    method = userargs.pop("method", "coordinate_ascent")
    template = userargs.pop("template", None)
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
            _flip_file(
                f, covs[template], out, n_embeddings, standardize, method, userargs
            )
    return dataset
