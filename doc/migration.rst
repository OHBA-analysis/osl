Migrating from osl-ephys 2
==========================

osl-ephys 3 is a rewrite of the pipeline around one config and one runner,
with the source reconstruction, parcellation and sign flipping shared with
osl-dynamics. It is not backwards compatible: the old entry points, script
names and module paths are gone rather than aliased. This page lists what
changed and what replaces it.

.. contents::
   :local:

One config, one runner
----------------------

``run_proc_chain`` / ``run_proc_batch`` (preprocessing) and
``run_src_chain`` / ``run_src_batch`` (source reconstruction) are replaced
by :func:`osl_ephys.pipeline.run_chain` (one session) and
:func:`osl_ephys.pipeline.run_batch` (a set of sessions in parallel). A
config can hold every stage:

.. code-block:: yaml

   maxfilter:
     - maxfilter: {scanner: Neo, mode: multistage, tsss: true}
   preproc:
     - filter: {l_freq: 1, h_freq: 45}
     - bad_segments: {picks: mag}
   coreg:
     - extract_surfaces: {include_nose: false}
     - extract_fiducials_and_headshape: {}
     - coregister: {use_nose: false}
   source_recon:
     - forward_model: {model: Single Layer}
     - beamform: {chantypes: [mag, grad], rank: {meg: 60}}
   parcellation:
     - parcellate: {parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz,
                    orthogonalisation: symmetric}
   group:
     - sign_flip: {n_embeddings: 15}

The v2 ``source_recon`` section is split into ``coreg`` (surfaces,
fiducials and headshape points, coregistration), ``source_recon`` (forward
model, beamformer or minimum norm) and ``parcellation``; ``group`` holds
the steps that run once over every session, and ``maxfilter`` runs
MaxFilter before preprocessing.

The sessions are a list of dicts, or a CSV file, with an ``id`` and the
files the steps need: ``raw_file``, ``smri_file``, ``preproc_file``,
``epoch_file``, ``pos_file``, ``elc_file`` and ``subject`` (sessions of a
subject share its surfaces). ``run_batch(config, sessions, outdir,
n_workers=4)`` replaces the dask client: there is no ``dask_client``
argument and dask is no longer a dependency.

The output layout
-----------------

::

   outdir/
     maxfilter/<id>_proc-tsss_meg.fif
     preproc/<id>_preproc-raw.fif   (+ _events.npy, _epo.fif, _ica.fif, ...)
     surfaces/<subject>/            the surfaces extracted from the MRI
     <id>/bem/, <id>/coreg/, <id>/src/
     <id>/<id>_parc-raw.fif         the parcel time courses
     <id>/<id>_sflip_parc-raw.fif   after sign flipping
     logs/<id>.log, logs/summary.csv
     qc/report.html                 the QC report

In v2 the preprocessed file was ``outdir/<id>/<id>_preproc-raw.fif`` and
the parcellated one ``outdir/<id>/parc/parc-raw.fif``.

Renamed steps and options
-------------------------

===================================  =====================================
v2                                   v3
===================================  =====================================
``compute_surfaces``                 ``extract_surfaces``
``extract_polhemus_from_info``,      ``extract_fiducials_and_headshape``
``extract_fiducials_from_fif``
``extract_polhemus_from_pos``        ``extract_fiducials_and_headshape_from_pos``
``extract_polhemus_from_elc``        ``extract_fiducials_and_headshape_from_elc``
``beamform``, ``freq_range``         ``beamform``, ``frequency_range``
``beamform_and_parcellate``          kept; or ``beamform`` then ``parcellate``
``minimum_norm_and_parcellate``      kept; or ``minimum_norm`` then
                                     ``parcellate: {source_method: minimum_norm}``
``fix_sign_ambiguity`` (per session)  ``sign_flip`` (group section; picks the
                                     template itself)
``allow_smri_scaling``               ``allow_mri_scaling``
``gridstep`` (surface source space)  ``spacing``
``apply_ica`` (manual ICA review)    ``apply_manual_ica``; MNE's apply is
                                     ``apply_ica``
``ica_autoreject``, ``ica_raw``      unchanged; a second ``ica_raw`` keeps
                                     both ICAs (``_ica.fif``, ``_ica-2.fif``)
===================================  =====================================

New steps: ``ica_label`` (automatic labelling with mne-icalabel),
``ica_ecg_eog_correlation``, ``decimate_headshape_points``, ``amm`` (OPM
interference rejection), ``maxfilter``. Options that no longer apply are
not accepted: an unknown option raises a ``TypeError`` from the function
that would have received it.

Data files
----------

The parcellations, masks and templates are downloaded from
`osl-files <https://github.com/OHBA-analysis/osl-files>`_ on first use
(``osl-ephys download-data`` fetches everything at once). The files have
new names, ``atlas-<Name>_nparc-<N>_space-MNI_res-<r>x<r>x<r>.nii.gz``:

===========================================================  ===========================================================
v2                                                           v3
===========================================================  ===========================================================
``Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz``      ``atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz``
``Glasser50_space-MNI152NLin6_res-8x8x8.nii.gz``             ``atlas-Glasser_nparc-50_space-MNI_res-8x8x8.nii.gz``
``dk_cortical.nii.gz``                                       ``atlas-DK_nparc-68_space-MNI_res-8x8x8.nii.gz``
``fMRI_parcellation_ds8mm.nii.gz``                           ``atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz``
``fMRI_parcellation_ds2mm.nii.gz``                           ``atlas-Giles_nparc-38_space-MNI_res-2x2x2.nii.gz``
``fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz``  ``atlas-Giles_nparc-39_space-MNI_res-8x8x8.nii.gz``
``fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz``  ``atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz``
``aal_cortical_merged_8mm_stacked.nii.gz``                   ``atlas-AAL_nparc-78_space-MNI_res-8x8x8.nii.gz``
``Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_4d_ds8.nii.gz``  ``atlas-Schaefer_nparc-100_space-MNI_res-8x8x8.nii.gz``
``HarvOxf-sub-Schaefer100-combined-2mm_4d_ds8.nii.gz``       ``atlas-SchaeferHarvOxf_nparc-114_space-MNI_res-8x8x8.nii.gz``
``HarvardOxford-sub-prob-bin-2mm.nii.gz``                    ``atlas-HarvOxfSub_nparc-21_space-MNI_res-2x2x2.nii.gz``
``dk_full.nii.gz``, ``giles_39_binary.nii.gz``, ...           ``legacy/<same name>``
===========================================================  ===========================================================

Commands
--------

The seven scripts are subcommands of one ``osl-ephys`` command:

=========================================  =============================================
v2                                         v3
=========================================  =============================================
``osl_preproc config.yml files.txt``       ``osl-ephys run config.yml sessions.csv outdir``
``osl_maxfilter files.txt outdir --tsss``  ``osl-ephys maxfilter files.txt outdir --tsss``
``osl_ica_label all outdir sub-01``        ``osl-ephys ica review outdir sub-01 --reject all``
``osl_ica_apply outdir``                   ``osl-ephys ica apply outdir``
``osl-ica-review 8000``                    ``osl-ephys ica review outdir --browser``
``osl-ica-apply ica_root raw_root``        ``osl-ephys ica apply outdir``
``osl_func module.function args``          ``osl-ephys func module.function args``
=========================================  =============================================

Reports
-------

The per-subject report pages (``subject_report.html``,
``summary_report.html``) are replaced by one ``qc/report.html`` for the
run: a table with one row per session (bad segments, bad channels, ICA
rejections, coregistration error, ...) that can be sorted and filtered,
next to the plots of the selected session. Each step saves its plots as
WebP files under ``qc/<section>/<id>/``, and custom figures put in
``dataset["fig"]`` get their own tab. :func:`osl_ephys.report.generate_report`
rebuilds the page from what has been saved.

Modules
-------

=====================================  =====================================
v2                                     v3
=====================================  =====================================
``osl_ephys.preprocessing``            ``osl_ephys.preproc`` (``functions``, ``steps``, ``dataset``, ``qc``, ``ica``)
``osl_ephys.source_recon.rhino``       ``osl_ephys.rhino``
``osl_ephys.source_recon.beamforming``  ``osl_ephys.source_recon.beamforming``
``osl_ephys.source_recon.parcellation``  ``osl_ephys.parcellation``
``osl_ephys.source_recon.sign_flipping``  ``osl_ephys.group.sign_flipping``
``osl_ephys.utils.spmio``, ``utils.opm``, ``utils.study``  ``osl_ephys.io``
``osl_ephys.report.preproc_report``    ``osl_ephys.preproc.qc`` and ``osl_ephys.report``
=====================================  =====================================

Custom functions are still ``function(dataset, userargs)`` passed as
``extra_funcs``; source-reconstruction custom functions take the dataset
too (``Session.from_dataset(dataset)`` gives the session's files) rather
than ``(outdir, subject, ...)``.

Removed
-------

- ``glm_irasa`` and the ``irasa`` option of the GLM spectrum (the sails
  implementation it wrapped no longer works).
- ``extract_rhino_files`` (migrated pre-2 derivatives), ``bem_display``
  (the forward model checks the BEM), ``apply_head2mni_xform`` (the forward
  model is on the MNI grid).
- The dask dependency and ``dask_client``.
