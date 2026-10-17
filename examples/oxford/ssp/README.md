# SSP pipeline

As the automated ICA pipeline, but the eye and heart artefacts are removed
with signal-space projection instead of ICA. You need the raw `.fif` file
of each session, with EOG and ECG channels, and a structural MRI of each
subject.

1. `1_maxfilter.py` runs MaxFilter (on a computer with a MaxFilter licence).
2. `2_preprocess.py` filters, downsamples and marks bad segments and
   channels, without ICA.
3. `3_ssp_denoise.py` computes one ECG and one EOG projector per sensor
   type from each session, applies them and overwrites the preprocessed
   file. The projectors and the power spectrum afterwards are plotted in
   `data/ssp`.
4. `4_coregister.py` extracts the surfaces from the MRI, coregisters the
   head to them and computes the forward model. Check the coregistration
   tab of `data/qc/report.html`; a session that looks off can be rerun
   with a higher `n_init`.
5. `5_source_reconstruct.py` beamforms and parcellates. The parcel time
   courses are `data/<id>/<id>_parc-raw.fif`.
6. `6_sign_flip.py` aligns the sign of each parcel across sessions. The
   output is `data/<id>/<id>_sflip_parc-raw.fif`.
