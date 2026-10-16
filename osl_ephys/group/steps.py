"""Sign flipping as config steps: ``fix_sign_ambiguity`` flips one session to
a template (a parcellation step) and ``sign_flip`` picks the template and
flips every session (the group step)."""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np

from osl_ephys.group import sign_flipping
from osl_ephys.pipeline.registry import step, ignored_options, renamed_option
from osl_ephys.session import Session

logger = logging.getLogger("osl_ephys")


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
    ignored_options(userargs, "n_init", "n_iter", "max_flips")
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
