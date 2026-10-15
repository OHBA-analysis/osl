# osl-ephys v3 refactor plan

Written 2026-10-15 from a survey of osl-ephys (main, v2.4.0), osl-dynamics
(v3.6.2+12, commit b64c6eb0) and osl-files, plus the original todo list and the
open issues/PRs on GitHub. Tracked on the "osl-ephys major overhaul" project
board and draft PR #453 (`v3` -> `main`).

## 1. The big picture

osl-dynamics already contains a rewritten M/EEG pipeline in `osl_dynamics.meeg`
(~12k lines): `maxfilter`, `amm` (OPM), `preproc`, `session`, `rhino`,
`source_recon` (LCMV + bilateral + virtual electrodes), `parcellation`,
`parallel`, `report`, plus `data.sign_flipping` and `files` (pooch download from
osl-files). It is released on PyPI and conda-forge (`osl-dynamics-base` has no
TensorFlow), uses pyproject + setuptools_scm, trusted publishing, codespell and
black. Most items on the todo list are "make osl-ephys look like that".

osl-ephys today (24k lines) is organised around a YAML config API
(`run_proc_batch` / `run_src_batch` and the `run_mne_*` / `run_osl_*` wrappers),
dask for parallelism, jinja2 per-subject HTML reports, bundled parcellations and
simulation data, setup.py packaging, manual releases, and a CI job pinned to
Python 3.8 with actions v2. Issue #449 reports the tests currently fail.

What osl-ephys has that osl-dynamics does not:

- the YAML config / wrapper API and `osl_preproc`, `osl_func`, `osl_maxfilter`,
  `osl_ica_label`, `osl_ica_apply` CLIs
- `glm` (sensor GLM spectrum / epochs via sails + glmtools)
- `plot_ica` interactive ICA labelling GUI
- `minimum_norm` (volumetric MNE; PR #445 fixes it) and `freesurfer_utils`
- `utils.spmio` (SPM MEEG I/O), `utils.study` (`Study` file finder), OPM
  conversion utilities, `utils.simulate` (MVAR simulation from a MEGIN template)
- the preproc report's richer plots (flowchart, ICA, artefact scan, digitisation,
  head movement, events, custom figures)
- EEG sensor moving example, CTF with sMRI fiducials example

## 2. Decisions to make before any code moves

These change what the later phases look like, so settle them first.

D1. Relationship with osl-dynamics (the one that matters most)

  Working decision (2026-10-15): port the `osl_dynamics.meeg` code into
  osl-ephys so that osl-ephys is a standalone package with the same
  dependency set as osl-dynamics minus TensorFlow. Section 8 lists what is
  ported, module by module.

  Why not depend on osl-dynamics instead: the package names say what the end
  state should be. osl-ephys is the M/EEG package and osl-dynamics is the
  dynamics package, so the dependency should eventually run the other way:
  once osl-ephys v3 is on conda-forge, osl-dynamics can drop its `meeg` copy
  and import from osl-ephys. Porting now is the first step of that; depending
  on osl-dynamics now would entrench the wrong direction.

  Consequences:
  - Until osl-dynamics switches over, the code exists in both packages. The
    `meeg` subpackage changed in 30 commits in the last 90 days, so the port
    should be done from a pinned osl-dynamics commit and recorded (section 8),
    and fixes made in one place after the port need a PR in the other until
    the switch.
  - Licences already match: osl-ephys has been Apache-2.0 since 2024-10-09
    (the `license='MIT'` in main's `setup.py` was stale metadata; `v3` has it
    right). Phase 1 aligns the rest with osl-dynamics: ship the full Apache
    text rather than the short header, add a NOTICE file (copyright line, OHBA
    attribution, the osl-files note), and set `license = "Apache-2.0"` with
    `license-files = ["LICENSE", "NOTICE"]` in `pyproject.toml`. Ported code
    then needs no extra attribution beyond the shared NOTICE.
  - The tests osl-ephys writes in phase 7 become the tests osl-dynamics has
    never had for this code, which is an argument for the switch happening
    soon after v3.

D2. Python floor: `>=3.10` (same as osl-dynamics; drops the 3.7-3.9
    classifiers and allows `X | None` annotations). Dependencies are the
    osl-dynamics set minus TensorFlow, plus what the osl-ephys-only modules
    need (section 8).

D3. Parallelism: drop dask and distributed, use the `multiprocessing` runner
    design (per-session log file, `summary.csv`, worker recycling, QC report
    hook). Config API keeps a single `n_workers` argument; `dask_client` goes
    with a clear error message for one release.

D4. Keep the YAML config API as the public face of osl-ephys, and make it
    one config for the whole pipeline. Today preprocessing and source
    reconstruction have separate config formats, separate `find_func`/
    `load_config` implementations and separate batch runners
    (`run_proc_batch`, `run_src_batch`). In v3 a single config can hold any
    mix of steps (preprocessing, surface extraction, coregistration, forward
    model, source reconstruction, parcellation, sign flipping) and one runner
    executes it per session. Running only the preprocessing section, or only
    the source-recon sections, is the same config with sections left out.
    It is what the workshops and the toolbox paper teach, and it is the thing
    osl-dynamics does not offer for M/EEG.

D5. Reports: replace the per-subject jinja2 pages with the osl-dynamics style
    (each step writes `summary.json` + WebP plots to a QC directory, one table
    report across sessions). Keep the osl-ephys-specific preproc plots as QC
    plots. `coreg.html` stays available as an option (the ported
    `plot_coregistration` writes it when asked); the PNG multi-view is the
    default, which is what issue #437 asks for.

D6. Version: the next release is v3.0.0 with a v2 to v3 migration page. v2.4.x
    stays installable for the toolbox paper and existing workshop material.

D7. Which of the open PRs and stale branches are merged into `v3`
    (section 5), and in what order.

D8. No feature is removed from osl-ephys unless there is a reason that is
    written down. The default disposition of every public function, class,
    wrapper, CLI and config option is "keep": either unchanged, or with a
    ported equivalent that does at least what the old one did. The feature
    inventory (section 9) lists every public name on `v3` with its
    disposition, and every "drop" carries its reason. A phase is not done
    while an entry in its scope still says "undecided".

## 3. Phases

Ordered so that each phase lands on a green CI and the dangerous ones (4, 5)
come after the scaffolding exists to test them.

### Phase 1: Packaging, tooling, CI  (todo: dependencies, codespell/black, release process)

- `pyproject.toml` with setuptools_scm; delete `setup.py`, `setup.cfg`,
  `requirements.txt`, `Makefile`; `__version__` from `importlib.metadata`.
  Version then lives only in git tags (fixes the four-file bump in
  `release_notes.md`; `CITATION.cff` version updated at release time only).
  The files already disagree: `setup.py` says 2.5.dev0, `__init__.py` says
  2.4.dev0, `conf.py` and `CITATION.cff` say 2.4.0.
- Quick correctness fixes to land before the rewrite so they are not lost:
  issue #446 (`set_format` undefined in `utils/logger.py:90`, `click` undefined
  in `preprocessing/batch.py:1094`), #440 (`detect_zeros` should skip EEG
  picks), and the inline TODO/XXX notes in `utils/file_handling.py:124`,
  `utils/spmio/spmmeeg.py:163`, `preprocessing/plot_ica.py` (five places),
  `source_recon/freesurfer_utils.py:89`, `source_recon/parcellation.py:205`,
  each either fixed or turned into a board item.
- Dependency audit (what is actually imported):
  - remove: `dask`, `distributed` (D3), `opencv-python` (one
    `morphologyEx` call in `rhino/surfaces.py`, replace with
    `scipy.ndimage`; issue #439), `open3d` soft import (issues #439, #193),
    `neurokit2`, `panel`, `h5io` (listed but never imported)
  - keep: `sails`, `glmtools` (GLM module; `glmtools` is NOT on conda-forge,
    see phase 9), `fslpy`, `tabulate`, `parse`, `numba`, `jinja2` (until D5)
  - add: `osl-dynamics` (D1a) or `pooch` (D1b/c)
  - pin `sails` or fix the incompatibility behind issue #449
- `.github/workflows/`: `tests.yml` (push + PR, matrix Python 3.10-3.13,
  ubuntu + macos, `pip install -e .[dev]`, pytest), `codespell.yml`,
  `black.yml`, `release.yml` (publish on GitHub release via PyPI trusted
  publishing; needs the `pypi` environment and the trusted publisher set up on
  pypi.org for osl-ephys). Delete `osl-cicd-actions.yml`.
- One-off black reformat in its own commit, listed in
  `.git-blame-ignore-revs`. Codespell config in `pyproject.toml`.
- Licence files aligned with osl-dynamics: full Apache-2.0 text in `LICENSE`,
  a `NOTICE` file, `license`/`license-files` in `pyproject.toml`, the
  classifier. `CITATION.cff` version (currently `2.5.0.dev1+semp.2`) set only
  at release time.
- `.readthedocs.yaml` to Python 3.12 and `doc/requirements.txt`; `html_logo`
  from `osl-files/docs/logo` (todo: logos).
- README: short description, badges, conda install first, "Releases" section
  copied from osl-dynamics; `release_notes.md` deleted; CONTRIBUTING trimmed
  (drop the git school, point at osl-dynamics' conventions).
- Remove import-time side effects in `osl_ephys/__init__.py` (NUMEXPR env var,
  logger level changes), the README-as-`__doc__` hack in every subpackage, and
  the module-level `matplotlib.use('Agg')` in `source_recon/wrappers.py` and
  `rhino/surfaces.py` (issues #450, #431). Set Agg only inside worker processes.

### Phase 2: Data to osl-files  (todo: test data, remove data, logos)

Reference files (8.4 MB in `source_recon/files`, 6.9 MB in
`utils/simulation_config`) move out; fetched on demand through a ported
`osl_ephys.files` (pooch, same cache directory as osl-dynamics).

Mapping to confirm (osl-files reprocessed everything onto the MNI152 grid, so
checksums will not match and some may be intentionally different).
`osl_dynamics/files/functions.py` already holds an old-to-new alias dict for
seven of these names; start from it:

| osl-ephys file | osl-files candidate |
| --- | --- |
| Glasser52_binary_space-MNI152NLin6_res-8x8x8 | atlas-Glasser_nparc-52_space-MNI_res-8x8x8 |
| Glasser50_space-MNI152NLin6_res-8x8x8 | atlas-Glasser_nparc-50_space-MNI_res-8x8x8 |
| reduced_hcp-mmp_2mm / _8mm | atlas-Glasser_nparc-52 at 2x2x2 / 8x8x8 ? |
| giles_39_binary | atlas-Giles_nparc-39_space-MNI_res-8x8x8 (confirm against the line above) |
| fmri_d100_parcellation_with_PCC_reduced_2mm (+ _ss5mm_ds8mm) | atlas-Giles_nparc-38 at 2x2x2 / 8x8x8 |
| fmri_d100_parcellation_with_PCC_tighterMay15_v2_{2mm,8mm,6mm_exclusive} | atlas-Giles_nparc-39 (8 mm; osl-dynamics maps the 8mm file to it) |
| aal_cortical_merged_8mm_stacked | atlas-AAL_nparc-78_space-MNI_res-8x8x8 |
| dk_full / dk_cortical | atlas-DK_nparc-68 (osl-dynamics maps dk_cortical to it); dk_full unmapped |
| Schaefer2018_100Parcels_7Networks_..._2mm_4d (+ _ds8) | atlas-Schaefer_nparc-100 at 8x8x8 (no 2 mm in osl-files) |
| HarvOxf-sub-Schaefer100-combined-2mm_4d (+ _ds8), HarvardOxford-sub-prob-bin-2mm | not in osl-files |
| fMRI_parcellation_ds2mm / ds8mm, WTA_fMRI_parcellation_ds2mm | not in osl-files |
| MNI152_T1_2mm_brain / 8mm_brain | mask/MNI152_T1_2mm_brain, 8mm_brain |
| *.xml label files | docs/parcellations in osl-files |

- Add the missing ones to osl-files (with licence files), or decide they are
  dropped. Keep an old-name to new-name alias dict in `parcellation.find_file`
  that warns for one release.
- `utils/simulation_config/*` (MEGIN template info fif, MVAR parameters) to
  osl-files `scanner/` or a new `simulation/` directory. `neuromag306_info.yml`
  and the `trees/` are tiny and can stay in the package.
- Add osl-ephys and its `osl-ephys-download-data` CLI to the "Packages that
  use these files" table in the osl-files README.
- Test data in osl-files under `testdata/{elekta,ctf,opm,eeg}/`:
  - non-identifiable: `mne.io.anonymize_info`, no sMRI (coreg tests use
    `surface/mni152` from osl-files, which also means no FSL is needed in CI),
    headshape/fiducials kept because coregistration tests need them
  - small: crop to tens of seconds, downsample, keep full channel set for
    Elekta (maxfilter / ICA-label paths depend on it)
  - features: a bad channel, a bad segment, ECG + EOG channels, a stim channel
    with events, head position channels (chpi) for the maxfilter code paths,
    CTF as a `.ds` directory plus `.pos`/`.elc`, multi-axis OPM (issue #427),
    EEG with a montage and a reference
  - osl-files is a plain git repo fetched over raw.githubusercontent, so keep
    each dataset to a few MB; larger tutorial datasets stay on OSF

### Phase 3: One config, one runner  (todo: refactor parallel processing, remove dask; unify preproc and source recon)

The unified config (D4):

```yaml
preproc:
  - filter: {l_freq: 1, h_freq: 45}
  - bad_segments: {picks: mag}
  - ica_label: {picks: meg}
surfaces:
  - extract_surfaces: {include_nose: false}
coreg:
  - extract_fiducials_and_headshape: {}
  - coregister: {use_nose: false, allow_mri_scaling: false}
forward_model:
  - forward_model: {model: Single Layer, gridstep: 8}
source_recon:
  - lcmv_beamformer: {chantypes: [mag, grad], rank: {meg: 60}}
parcellation:
  - parcellate: {parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz,
                 orthogonalisation: symmetric}
```

- One step registry replaces the two `find_func` implementations: every
  wrapper registers under a section name, and custom functions are passed in
  the same way for every section (today `extra_funcs` only exists for
  preprocessing). A step's wrapper receives the session dict and the loaded
  data object and returns them, so sections chain without going through the
  filesystem unless the user asks for intermediate files.
- One session manifest (list of dicts, or a CSV, with `id`, `raw_file`,
  `smri_file`, `subject`, optional `pos_file`/`elc_file`) feeds every section;
  the `Session` object from phase 4 is built from it. `utils.file_handling`'s
  glob-based input handling and `Study` become helpers for building the
  manifest, not part of the runner.
- Section-level granularity: `surfaces` runs once per subject rather than per
  session (the runner groups by `subject`), the other sections run per
  session. A config with only some sections present skips the others and
  checks (via `check_up_to_date`) that the earlier outputs it needs exist.
- Group-level steps (sign flipping template selection, `fix_sign_ambiguity`)
  run after the per-session loop as a `group` section, which also answers
  issue #383.
- `run_batch(config, manifest, outdir, n_workers=N)` is the single entry
  point, with `run_chain` for one session; `run_proc_batch` and
  `run_src_batch` remain for one release as thin wrappers that build the
  equivalent config and warn.
- `osl_preproc` CLI becomes `osl-ephys run config.yml manifest.csv outdir`
  (the old name kept as an alias for one release).

Parallelism underneath it:


- One runner (ported from `osl_dynamics.meeg.parallel.run`)
  used by both `run_proc_batch` and `run_src_batch`: items are session dicts,
  the runner handles per-session logging, failure isolation, `summary.csv`,
  and calls report generation at the end.
- `run_proc_batch(... n_workers=N)` replaces `dask_client=True` plus a
  user-created `Client`. `utils.parallel.dask_parallel_bag`, `test_parallel.py`
  and `examples/parallelisation/` rewritten.
- `utils.logger` replaced by the per-session logger the runner expects
  (issue #323).

### Phase 4: Source reconstruction and parcellation  (todo: match osl-dynamics)

- Port (see the inventory above) `Session` (fixed output layout: `{outdir}/{id}/{bem,coreg,src}`),
  `check_up_to_date`, `save_params` JSON next to each output, the consolidated
  `rhino` API (surfaces, fiducials from fif/pos/elc, coregistration, forward
  model), `source_recon.lcmv_beamformer` (with bilateral pairs, which
  supersedes PR #426 and issue #103), `parcellation` (Parcellation class,
  `parcellate_lcmv`, symmetric orthogonalisation, `save_as_fif`), and the
  osl-dynamics sign flipping (compare with the `new-sign-flip` branch first).
- Keep and port onto the new Session: `minimum_norm` (merge PR #445 first),
  `freesurfer_utils`, `nii.py`, `extract_rhino_files`, EEG sensor moving.
- Rewrite `source_recon/wrappers.py` so each YAML step is a thin call into the
  new functions, registered in the phase 3 step registry under the `surfaces`,
  `coreg`, `forward_model`, `source_recon` and `parcellation` sections; old
  config keys map to new ones with warnings where cheap. Output layout
  changes, so the migration page documents old vs new paths.
- Closes along the way: #441/#442 (transform_vtk_mesh), #409 (coreg float),
  #418 (nilearn private function), #393 (fix_sign_ambiguities), #437
  (coreg.html), #439/#193 (open3d, opencv), #200 (MNE option exists), #383
  (group steps after individual steps become a documented pattern).
- Retire what the new code replaces (`rhino/utils.py`, `rhino/coreg.py`,
  `rhino/surfaces.py`, `rhino/polhemus.py`, `rhino/forward_model.py`,
  `beamforming.py`, most of `parcellation.py`) only once every public name in
  them has a disposition in the feature inventory (section 9). Functions the
  osl-dynamics code has no equivalent for (e.g. `delete_headshape_points`,
  `plot_polhemus_points`, `fsleyes` helpers, `get_leadfields`,
  `voxel_timeseries`, `transform_recon_timeseries`, `recon_timeseries2niftii`)
  are moved, not dropped.

### Phase 5: Reporting  (todo: update reporting)

- Port `osl_dynamics.meeg.report`: each step writes `summary.json` and WebP
  plots into `qc_dir/<step>/<id>/`; `generate_report` builds one sortable table. The preproc plots
  that only osl-ephys has (flowchart, ICA components, artefact scan,
  digitisation, head movement, events, custom figures) become QC plot
  functions in `preprocessing`. `report/templates/*.html`, `preproc_report.py`
  (1745 lines) and `src_report.py` go.
- `osl_ica_label` keeps working from the saved fif + ICA files; its GUI
  (`plot_ica.py`) is untouched by this phase.
- Closes #411, #413 (report robustness), #135 (sequential ICA overwrites),
  #212, #134, #170, #251 are triaged: implement in the new report or close.

### Phase 6: Preprocessing cleanup

- Merge PR #444 (GESD) and fix #440 (maxfilt zeros fallback) before touching
  `osl_wrappers.py`, so the fixes are not lost in the rewrite.
- Port `osl_dynamics.meeg.preproc` and add wrappers for its functions
  (`detect_bad_segments`, `detect_bad_channels`, `ica_label` via mne-icalabel /
  MEGnet) alongside the existing `run_osl_*`; issue #427 (multi-axis OPM) and
  #373 (rectified signal warning) addressed here.
- `maxfilter`: adopt the osl-dynamics rewrite (chunking, CBU three-stage,
  split-file chains; fixes #192) behind the existing `osl_maxfilter` CLI.
- `glm`: keep as is; only the sails/glmtools pin changes.
- `utils.spmio`, `utils.study`, `utils.opm`: keep; `utils.simulate` reads the
  template from osl-files.

### Phase 7: Tests and GitHub Actions  (todo: tests)

- Session-scoped fixtures that fetch the osl-files test data once per CI run
  (cache `~/.cache/osl-files` with `actions/cache`).
- Test the things that break silently today: config loading and version
  checks, every `run_mne_*`/`run_osl_*` wrapper on each modality's test data,
  the chain and batch on two sessions with `n_workers=2`, the parallel runner's
  failure isolation and summary, parcellation maths (orthogonalisation,
  parcel weights), sign flipping on synthetic data, coreg + forward + LCMV +
  parcellate end to end on MNI152 surfaces, report generation, every CLI's
  `--help`, and the data fetch alias table.
- FSL-dependent code (`extract_surfaces`, FNIRT) marked `fsl` and skipped
  unless `FSLDIR` is set. MaxFilter can only be tested at the command-building
  level.
- `osl_ephys.check_installation()` / `osl-ephys-check` CLI (issue #425)
  replaces `utils.package.run_package_tests`.

### Phase 8: Examples and docs  (todo: update examples)

- `examples/{elekta,ctf,eeg,opm,parallelisation}` rewritten against the v3
  API, mirroring the osl-dynamics `meg_preproc` script layout (config block at
  the top, `process_session`, `parallel.run`). `examples/toolbox-paper` and
  `examples/oxford` are frozen with a note that they target v2.4.
- Tutorials: keep the OSF datasets for the long ones; the short ones can run
  on the osl-files test data so the docs build exercises the code.
- Docs: install page in the osl-dynamics style (conda-forge first), migration
  page, `llms.txt`, FAQ refresh; issues #352, #361, #386.

### Phase 9: conda-forge and mne-installers  (todo: repackage for conda, mne-installers)

- Blocker: `glmtools` has no conda-forge feedstock. Either submit one (it is a
  pure-Python package, so a `noarch` recipe is small) or make `glm` an optional
  extra (`osl-ephys[glm]`) so the base package can go to conda-forge now.
- Recipe modelled on `osl-dynamics-feedstock/recipe/recipe.yaml`: `noarch:
  python`, entry points for the five CLIs, import tests per subpackage,
  `vtk * *osmesa*` on linux-64 so the 3D coreg plot renders headless (this is
  the "make sure the coreg plotting works" item; verify with a headless test
  that calls `plot_coregistration` with `pyvista.OFF_SCREEN = True`).
- mne-installers: a PR adding `osl-ephys =X.Y.Z` (and `osl-dynamics`, which
  is not there either) to `recipes/mne-python/construct.yaml`. Requires the
  conda-forge package first and that every dependency resolves with the
  installer's pinned MNE and Qt stack; check `pyside6`/`pyvistaqt` versions
  against theirs before opening the PR.

### Phase 10: Release v3.0.0

- Changelog from the project board; migration page (config key renames,
  output layout, dask removal, dropped parcellation names, dropped
  `coreg.html`); tag `v3.0.0`, the release workflow publishes to PyPI, the
  conda-forge bot opens the version bump.

### Port inventory: what comes from osl-dynamics

Everything below is in osl-dynamics at commit b64c6eb0 (2026-10-12). "Port"
means copy into osl-ephys with the import paths changed and the module
docstrings pointing at osl-ephys; "replace" names the osl-ephys code it makes
redundant. Phase numbers say when.

| osl-dynamics | Lines | Into osl-ephys | Replaces | Phase |
| --- | --- | --- | --- | --- |
| `files/_fetch.py`, `files/__init__.py`, `files/functions.py`, `files/{parcellation,mask,surface,mni152_surfaces,scanner,scene}` | ~400 | `osl_ephys/files/` with an `osl-ephys-download-data` CLI; same cache directory so both packages share downloads | `source_recon/files/`, `utils/simulation_config/`, `parcellation.find_file` | 2 |
| `utils/logger.py` (`MEEGSessionLogger`, `set_logging_level`) | 130 | `utils/logger.py` | current `utils/logger.py` (`log_or_print`, `set_up`) | 3 |
| `meeg/parallel.py` (`run`, `log`, `warning`, thread limiting, `summary.csv`) | 283 | `utils/parallel.py` | `dask_parallel_bag`, the dask paths in both batch functions | 3 |
| `meeg/session.py` (`Session`, `SurfaceFilenames`, `CoregFilenames`, `check_up_to_date`, `save_params`) | 322 | `source_recon/session.py` | `rhino.get_surfaces_filenames`, `get_coreg_filenames`, the ad hoc path building in `wrappers.py` | 4 |
| `meeg/rhino.py` (surfaces, fiducials from fif/pos/elc, stray point removal, coregistration + QC, BEM repair, forward model, MNI grid, FNIRT) | 3860 | `source_recon/rhino.py` (one module; the `rhino/` package goes) | `rhino/utils.py`, `coreg.py`, `surfaces.py`, `polhemus.py`, `forward_model.py`, `fsl_utils.py` | 4 |
| `meeg/source_recon.py` (LCMV, bilateral pairs, virtual electrodes, QC plot) | 1760 | `source_recon/beamforming.py` | current `beamforming.py` and the #426 implementation merged into v3 | 4 |
| `meeg/parcellation.py` (`Parcellation`, `parcellate`, `parcellate_lcmv`, orthogonalisation, `save_as_fif`, `convert_to_mne_raw`, QC plots) | 1118 | `source_recon/parcellation.py` | current `parcellation.py` | 4 |
| `data/sign_flipping.py` + the two helpers it uses from `data/processing.py` (`standardize`, `time_embed`) | 430 + 60 | `source_recon/sign_flipping.py` | current `sign_flipping.py` (compare with the `new-sign-flip` branch first) | 4 |
| `utils/plotting.py` `plot_psd_topo` and `plot_brain_surface`, `analysis/power.py` pieces used by `parcellation.save_qc_plots` | ~300 of 2742 | `utils/plotting.py` (only the functions needed) | nothing | 4 |
| `meeg/preproc.py` (`detect_bad_segments`, `detect_bad_channels`, GESD, `ica_label`, `ica_ecg_eog_correlation`, `decimate_headshape_points`, QC plots) | 1323 | `preprocessing/functions.py`, wrapped by `run_osl_*` steps | `osl_wrappers.gesd`, `bad_segments`, `bad_channels`, `detect_artefacts`; `ica_label` gains the mne-icalabel route (#427) | 6 |
| `meeg/maxfilter.py` (chunking, multistage, CBU three-stage, split chains, bad-channel merging) | 1482 | `maxfilter/maxfilter.py` behind the `osl_maxfilter` CLI (osl-dynamics has no CLI; keep ours) | current `maxfilter.py` (fixes #192) | 6 |
| `meeg/amm.py` (`apply_amm`, OPM interference rejection) | 1196 | `preprocessing/amm.py` + a `run_osl_amm` step | nothing (new capability) | 6 |
| `meeg/report.py` (`generate_report`, `TABS`, `CAPTIONS`) | 598 | `report/report.py` | `preproc_report.py`, `src_report.py`, `report/templates/` | 5 |
| `pyproject.toml` structure, `.github/workflows/{release,codespell,black}.yml`, `doc/requirements.txt`, README "Releases" section, `conf.py` logo and autoapi settings | | same paths | `setup.py`, `setup.cfg`, `requirements.txt`, `Makefile`, `release_notes.md`, `osl-cicd-actions.yml` | 1 |
| `examples/meg_preproc/` layout (config block, `process_session`, `parallel.run`) and tutorials 0-1, 0-3, 0-4, 0-5 | | `examples/` and `doc/source/tutorials/` adapted to the config API | current examples | 8 |
| `__init__.py` (version from `importlib.metadata`, one `logging.basicConfig`) | 20 | `osl_ephys/__init__.py` | the NUMEXPR and logger-level side effects | 1 |

Not ported, because osl-ephys has nothing to gain or already has it:
`config_api` (osl-dynamics' YAML pipeline is for dynamics models, not M/EEG
steps; osl-ephys keeps its own config API and extends it in phase 3),
`models`, `inference`, `analysis` (apart from the plotting pieces above),
`simulation`, `data.base`.

Kept from osl-ephys because osl-dynamics has no equivalent: the config API
and all CLIs, `glm`, `plot_ica` GUI and `osl_ica_label`/`osl_ica_apply`,
`minimum_norm`, `freesurfer_utils`, `nii.py`, `utils.spmio`, `utils.study`,
`utils.opm`, `utils.simulate`, the SEMP code from #452, and the preproc QC
plots that osl-dynamics lacks (flowchart, artefact scan, digitisation, head
movement, events, custom figures).

Dependencies after the port (osl-dynamics' list minus TensorFlow, minus
what only the dynamics models and analyses use, plus osl-ephys-only needs):

| Keep from osl-dynamics | Used by |
| --- | --- |
| numpy, scipy, matplotlib, pandas, pyyaml, packaging | everywhere |
| mne, mne-icalabel[onnx] | preprocessing, rhino, beamforming |
| nibabel, nilearn, fslpy, scikit-image, scikit-learn, numba | rhino, parcellation |
| pyvista, pyvistaqt, pyside6, trame, trame-vtk, trame-vuetify | coregistration 3D plot (off-screen and interactive) |
| pooch | files |
| h5io | LCMV filters file (`filters-lcmv.h5`) |

| Add for osl-ephys-only code | Used by |
| --- | --- |
| sails, glmtools | glm (make `glm` an optional extra if glmtools stays off conda-forge) |
| jinja2 | until phase 5 removes the templates |
| parse | `utils.study` |

| Drop from osl-dynamics' list | Reason |
| --- | --- |
| tensorflow, tensorflow-probability, tf-keras | models only |
| mat73, osfclient, ipyevents, ipywidgets, nest-asyncio2, tqdm, pqdm, threadpoolctl, seaborn, h5py | dynamics data loading, notebooks, model training |

| Drop from osl-ephys' current list | Reason |
| --- | --- |
| dask, distributed | replaced by the multiprocessing runner |
| opencv-python, open3d | one morphology call (scipy.ndimage) and a soft import |
| neurokit2, panel, tabulate | never imported, or only by the reports that phase 5 removes |

## 4. Suggested order and milestones

1. Decide D1-D7 (this week).
2. Phase 1 on a `v3` branch with CI green on the existing tests (fix #449 to
   get there). Merge. Tag nothing yet.
3. Phase 2 (data) and phase 7's fixture scaffolding together, because the new
   tests need the test data and the data move needs tests to prove the aliases.
4. Phase 3 (config + runner, first with only the `preproc` section wired up),
   then phase 4 (source recon sections) with its tests, then phase 5
   (reports), then phase 6 (preproc). Each as its own PR against `v3`.
5. Phases 8 and 9 in parallel once the API is stable.
6. Phase 10.

Everything up to step 2 is low risk and useful even if the later phases slip.

## 5. Merge the open PRs into the v3 branch

The five open PRs are merged into the `v3` branch before the rewrite starts,
so that their fixes are ported along with everything else rather than lost.
Order them smallest first so conflicts stay local:

| PR | Size | Notes |
| --- | --- | --- |
| #442 transform_vtk_mesh fix | small | Closes #441. The function is replaced in phase 4, but the fix keeps v3 working until then. |
| #444 GESD fixes | small, no checks ran | Goes into `osl_wrappers.py` ahead of the phase 6 rewrite. |
| #445 volumetric minimum norm fix | small, checks failing | Fix the checks on the branch first; phase 4 keeps minimum norm. |
| #426 bilateral beamformer | +1745 / -844, checks failing | Merge for its tests and QC plots; the implementation is replaced by the osl-dynamics bilateral pairs in phase 4. Decide whether that is worth the conflicts or whether a pointer is enough. |
| #452 SEMP (EEG-fMRI toolbox, manual ICA browser, bug fixes) | +9749 / -99, 56 files, checks pass | Largest and last. Decide whether SEMP is in scope for v3; if not, cherry-pick the bug fixes and the ICA browser. |

None of the five has had a review yet, so each merge is also its review.

The seven stale branches (`bilateral_beamformer`, `leadfields`,
`new-sign-flip`, `semp`, `viz`, `vol-mne`, `ohba-examples`) each need an
owner to say merge, port or delete before `v3` diverges. `new-sign-flip` is
the one to look at before phase 4 picks a sign-flipping implementation.

## 6. Risks

- Scope: phases 4 and 5 remove roughly 9k lines and change output paths;
  workshop material and existing derivatives directories depend on the old
  layout. The migration page and a v2.4.x maintenance tag are the mitigation.
- Two copies of the meeg code until osl-dynamics switches to importing from
  osl-ephys. Record the ported commit, and treat any fix to one copy as a PR
  to the other until then.
- osl-files size: test recordings in a git repo are permanent history. Agree
  a per-dataset budget before adding anything.
- `glmtools` on conda-forge is outside the team's control unless someone on the
  team becomes its maintainer.

## 7. Tracking on the project board

No new GitHub issues for tracking. The board (project 4, "osl-ephys major
overhaul") holds:

- the 5 open PRs, in "In progress", because they are merged into `v3`
- one draft item "v3 design decisions" for D1-D7
- one draft item per phase
- one draft item for merging the PRs and settling the stale branches
- one draft item "Revisit the open issues after v3"

The 48 open issues are deliberately not on the board. Many will be closed or
made irrelevant by the rewrite, so they are triaged once v3 exists; the
"closes #..." notes in the phases above are hints for that pass, not
commitments.

The script that does this is ~/Desktop/osl-ephys-populate-board.sh
(project id PVT_kwDOAOityM4BnKN4, Status field PVTSSF_lADOAOityM4BnKN4zhlwG7M,
options Todo=f75ad846, In progress=47fc9ee4, Done=98236657). Suggested
Priority options once someone creates them: P0 (blocks a release), P1 (in the
v3 scope), P2 (nice to have).

## 9. Feature inventory (D8)

Every public name on `v3` (commit 01e83fe), grouped by module, with its v3
disposition. "Keep" = unchanged or moved; "Port" = replaced by the ported
osl-dynamics equivalent, which must cover the same use; "Drop" = removed, with
the reason. Update this table as phases land; a phase is not done while a
line in its scope says "undecided".

| Module | Public names | Disposition |
| --- | --- | --- |
| CLIs | `osl_maxfilter`, `osl_ica_label`, `osl_ica_apply`, `osl_preproc`, `osl_func`, `osl-ica-review`, `osl-ica-apply` | Keep all. `osl_preproc` gains `osl-ephys run` as the new name and stays as an alias for one release. Decide whether the two ICA review/apply pairs (ica_label vs manual_ica from #452) merge into one. |
| `preprocessing.batch` | `run_proc_chain`, `run_proc_batch`, `load_config`, `find_func`, `check_config_versions`, `get_config_from_fif`, `append_preproc_info`, `import_data`, `read_dataset`, `write_dataset`, `plot_preproc_flowchart`, `print_custom_func_info` | Keep; `run_proc_batch` becomes a wrapper over `run_batch` (phase 3), `find_func`/`load_config` become the shared registry. Config provenance in the fif (`get_config_from_fif`, `append_preproc_info`) is kept. |
| `preprocessing.mne_wrappers` | 23 `run_mne_*` steps | Keep all. |
| `preprocessing.osl_wrappers` | `gesd`, `detect_artefacts`, `detect_maxfilt_zeros`, `bad_segments`, `bad_channels`, `drop_bad_epochs`, `detect_bad_channels_psd`, `run_osl_read_dataset`, `run_osl_bad_segments`, `run_osl_bad_channels`, `run_osl_drop_bad_epochs`, `run_osl_zscore_present_data`, `run_osl_glm_*` (4), `run_osl_manual_ica`, `run_osl_apply_ica` | Keep all wrappers. `gesd`/`bad_segments`/`bad_channels` are Port (osl-dynamics `detect_bad_segments`/`detect_bad_channels`, same GESD, with the #444 fixes carried over); `detect_maxfilt_zeros`, `drop_bad_epochs`, `detect_bad_channels_psd`, zscore and GLM steps have no osl-dynamics equivalent and are Keep. |
| `preprocessing.ica_label`, `plot_ica`, `manual_ica.*` | `ica_label`, `apply`, `plot_ica`, `osl_MNEBrowseFigure`, `manual_ica`, `apply_ica`, review server | Keep (GUI and manual workflow; nothing in osl-dynamics). Add the mne-icalabel automatic route from osl-dynamics as a new wrapper. |
| `preprocessing.semp.*` | AAS/OBS/SSP epoching, metrics, plots, tracer | Keep (merged from #452); scope of maintenance to be agreed with its author. |
| `maxfilter` | `run_maxfilter`, `run_multistage_maxfilter`, `run_cbu_3stage_maxfilter`, `run_maxfilter_batch`, `quick_load_dig`, `fit_cbu_origin`, `main` | Port (osl-dynamics `run_maxfilter` covers standard, multistage, CBU 3-stage, chunking, split chains) behind the same CLI; `run_maxfilter_batch` keeps its file-list interface. |
| `glm.*` | `glm_spectrum`, `glm_irasa`, `glm_epochs`, group variants, permutation classes, readers, all `plot_*` | Keep all (osl-dynamics has no sensor GLM). |
| `report.preproc_report` | `gen_report_from_fif`, `gen_html_*`, `plot_flowchart`, `plot_rawdata`, `plot_channel_time_series`, `plot_sensors`, `plot_channel_dists`, `plot_spectra`, `plot_freqbands`, `plot_digitisation_2d`, `plot_headmovement`, `plot_eog_summary`, `plot_ecg_summary`, `plot_bad_ica`, `plot_summary_bad_segs`, `plot_summary_bad_chans`, `plot_artefact_scan`, `print_scan_summary`, `plot_events`, `plot_custom_figures` | Every `plot_*` is Keep (becomes a QC plot in the new report). `gen_html_*` and the per-subject pages are Drop: replaced by the table report, which shows the same plots per selected session. `gen_report_from_fif` is Keep as the entry point that regenerates QC plots from a saved fif. |
| `report.src_report` | `gen_html_*`, `plot_config`, `plot_parcellation`, `plot_sign_flipping_results`, `add_to_data`, `update_config` | `plot_*` Keep as QC plots; `gen_html_*`/`add_to_data`/`update_config` Drop (replaced by `summary.json` per step and the table report). |
| `source_recon.batch` | `run_src_chain`, `run_src_batch`, `load_config`, `find_func` | Keep as wrappers over the phase 3 runner/registry. |
| `source_recon.wrappers` | `extract_polhemus_from_info`, `extract_fiducials_from_fif`, `extract_polhemus_from_pos`, `extract_polhemus_from_elc`, `remove_stray_headshape_points`, `save_mni_fiducials`, `rescale_sensor_positions`, `compute_surfaces`, `make_watershed_bem`, `coregister`, `forward_model`, `beamform`, `minimum_norm`, `parcellate`, `beamform_and_parcellate`, `minimum_norm_and_parcellate`, `find_template_subject`, `fix_sign_ambiguity`, `extract_rhino_files` | Keep every config step name. Bodies become calls into the ported rhino/beamforming/parcellation; `minimum_norm*`, `make_watershed_bem`, `rescale_sensor_positions`, `extract_rhino_files` keep their osl-ephys implementations. |
| `source_recon.rhino.coreg` | `coreg`, `coreg_metrics`, `coreg_display`, `bem_display`, `apply_head2mni_xform`, `get_coreg_filenames` | Port (`coregister_head_and_mri`, `coreg_error`/`coreg_rms`, `plot_coregistration`, `CoregFilenames`). `bem_display` and `apply_head2mni_xform`: undecided (check osl-dynamics covers them). |
| `source_recon.rhino.surfaces` | `compute_surfaces`, `surfaces_display`, `plot_surfaces`, `check_if_already_computed`, `get_surfaces_filenames` | Port (`extract_surfaces`, `plot_surfaces`, `SurfaceFilenames`, `check_up_to_date`). `surfaces_display` (fsleyes): Keep. |
| `source_recon.rhino.forward_model` | `forward_model`, `make_fwd_solution`, `setup_volume_source_space` | Port. |
| `source_recon.rhino.polhemus` | `get_polhemus_from_info`, `extract_polhemus_from_info`, `save_mni_fiducials`, `plot_polhemus_points`, `delete_headshape_points`, `remove_stray_headshape_points`, `extract_polhemus_from_pos`, `extract_polhemus_from_elc`, `rescale_sensor_positions` | Port the extract/remove functions; Keep `plot_polhemus_points`, `delete_headshape_points` (interactive), `save_mni_fiducials`, `rescale_sensor_positions` (no osl-dynamics equivalent). |
| `source_recon.rhino.fsl_utils` | `setup_fsl`, `check_fsl`, `fsleyes`, `fsleyes_overlay` | Keep. |
| `source_recon.rhino.utils` | 35 names: geometry (`icp`, `rhino_icp`, `rigid_transform_3D`, `best_fit_transform`, `xform_points`, `nearest_neighbor`), nifti helpers (`niimask2*pointcloud`, `get_vol_info_from_nii`, `get_sform`, `get_orient`, `check_nii_for_nan`, `timeseries2nii`, `recon_timeseries2niftii`), mesh helpers (`get_vtk_mesh_native`, `transform_vtk_mesh`, `create_freesurfer_mesh*`, `transform_bet_surfaces`), FLIRT/MNE transforms, `extract_rhino_files`, headshape helpers (`save_polhemus_fif`, `downsample_headshape`, `rotate_pointcloud`, `grid_average_downsample`, `replace_headshape`), `save_or_show_renderer`, `system_call`, `get_gridstep`, `majority`/`binary_majority3d`, `closest_node` | Port the ones the ported rhino has privately; Keep as public helpers the ones users call directly: `extract_rhino_files`, `recon_timeseries2niftii`, `timeseries2nii`, `get_vol_info_from_nii`, `downsample_headshape`/`replace_headshape`/`save_polhemus_fif` (osl-dynamics `decimate_headshape_points` covers the first), `fsleyes` display helpers. Each of the 35 gets its own line before phase 4 closes. |
| `source_recon.beamforming` | `make_lcmv`, `apply_lcmv`, `apply_lcmv_raw`, `load_lcmv`, `get_recon_timeseries`, `transform_recon_timeseries`, `get_lcmv_weights`, `get_leadfields`, `plot_dipole_locations`, `voxel_timeseries`, `make_plots`, `get_beamforming_filenames` | Port `make_lcmv`/`apply_lcmv*`/`load_lcmv` (`lcmv_beamformer`, `apply_lcmv_beamformer`, `filters-lcmv.h5`); Keep `get_recon_timeseries`, `transform_recon_timeseries`, `get_lcmv_weights`, `get_leadfields` (branch `leadfields`), `plot_dipole_locations`, `voxel_timeseries` on top of the ported filters. |
| `source_recon.minimum_norm` | `create_inverse_operator`, `apply_inverse_operator_surf`, `apply_inverse_operator_vol`, `calc_noise_cov`, `morph_surface`, `get_mne_filenames` | Keep (with #445). |
| `source_recon.freesurfer_utils` | `setup_freesurfer`, `check_freesurfer`, `recon_all`, `make_watershed_bem`, `make_fsaverage_src`, filename helpers | Keep. |
| `source_recon.parcellation` | `load_parcellation`, `load_parcellation_description`, `guess_parcellation`, `vol_parcellate_timeseries`, `surf_parcellate_timeseries`, `resample_parcellation`, `local_orthogonalise`, `symmetric_orthogonalise`, `parcel_centers`, `plot_parcellation`, `plot_psd`, `plot_correlation`, `convert2niftii`, `convert2mne_raw`, `convert2mne_epochs`, `spatial_dist_adjacency`, `parcel_vector_to_voxel_grid`, `convert2source_estimate`, `plot_source_topo`, `assign_voxels_to_binary_parcellation` | Port `vol_parcellate_timeseries`, `symmetric_orthogonalise`, `convert2mne_raw`, `convert2niftii`, `parcel_vector_to_voxel_grid`, `plot_parcellation`, `plot_psd`; Keep `surf_parcellate_timeseries`, `convert2source_estimate`, `convert2mne_epochs`, `local_orthogonalise`, `resample_parcellation`, `spatial_dist_adjacency`, `plot_correlation`, `plot_source_topo`, `guess_parcellation`, `load_parcellation_description`, `assign_voxels_to_binary_parcellation`. |
| `source_recon.sign_flipping` | `find_flips`, `find_template_subject`, `load_covariances`, `covariance_matrix_correlation`, `randomly_flip`, `apply_flips_to_covariance`, `apply_flips`, `time_embed`, `std_data` | Port (osl-dynamics `sign_flipping` + `new-sign-flip` branch comparison); `find_template_subject` and the config step names Keep. |
| `source_recon.nii`, `source_recon.utils` | `convert_4dparc_to_3d`, `convert_3dparc_to_4d`, `spatially_downsample`, `append_4d_parcellation`, `find_file` | Keep (`find_file` becomes the osl-files lookup with the alias table). |
| `utils.file_handling`, `utils.study` | `process_file_inputs`, `find_run_id`, `validate_outdir`, `get_rawdir`, `add_subdir`, `sanitise_filepath`, `osl_print`, `Study` | Keep; they build the session manifest. |
| `utils.logger` | `set_up`, `set_level`, `get_level`, `log_or_print` | Port (`MEEGSessionLogger`, `set_logging_level`); `log_or_print` kept as a thin shim since wrappers call it everywhere. |
| `utils.parallel` | `dask_parallel_bag` | Drop: replaced by the multiprocessing runner (D3). |
| `utils.simulate`, `utils.simulation_config` | `simulate_data`, `simulate_raw_from_template`, `simulate_rest_mvar` | Keep; template and MVAR parameters come from osl-files. The duplicate `simulation_config/simulate.py` module is Drop (same functions as `utils.simulate`). |
| `utils.spmio` | `SPMMEEG`, `Data`, `Channel`, `Montage`, `Trial`, `Event`, helpers | Keep. |
| `utils.opm` | `convert_notts`, `correct_mri` | Keep. |
| `utils.misc`, `utils.package`, `utils.version_utils`, `utils.run_func` | `ensure_dir`, `proc_userargs`, `mean_psd_in_band`, `set_random_seed`, `soft_import`, `run_package_tests`, `check_version`, `osl_func` main | Keep; `run_package_tests` becomes the `osl-ephys-check` CLI (#425). |
| Package data | `source_recon/files/*`, `utils/simulation_config/*`, `neuromag306_info.yml`, `trees/*.tree` | Move to osl-files (phase 2) with old-name aliases; `trees` and the yml stay in the package. |
