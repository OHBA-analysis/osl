"""The whole pipeline in one run, without the QC report and its plots.

MaxFilter, preprocessing with automatic ICA labelling, coregistration,
forward model, beamforming, parcellation and sign flipping, for a computer
where plotting fails. qc_dir=False turns the report off; the logs are
still in data/logs. Needs a MaxFilter licence for the first section.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
maxfilter:
  - maxfilter: {maxpath: /neuro/bin/util/maxfilter, scanner: Neo, mode: multistage, tsss: true, headpos: true, movecomp: true}
preproc:
  - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
  - notch_filter: {freqs: 50 100}
  - resample: {sfreq: 250}
  - bad_segments: {segment_len: 500, picks: mag}
  - bad_segments: {segment_len: 500, picks: grad}
  - bad_segments: {segment_len: 500, picks: mag, mode: diff}
  - bad_segments: {segment_len: 500, picks: grad, mode: diff}
  - bad_channels: {picks: mag}
  - bad_channels: {picks: grad}
  - ica_raw: {picks: meg, n_components: 40}
  - ica_autoreject: {ecgmethod: correlation, eogthreshold: auto}
  - interpolate_bads: {}
coreg:
  - extract_surfaces: {include_nose: false}
  - extract_fiducials_and_headshape: {}
  - remove_stray_headshape_points: {}
  - coregister: {use_nose: false, use_headshape: true, n_init: 1}
source_recon:
  - forward_model: {model: Single Layer}
  - beamform: {chantypes: [mag, grad], rank: {meg: 60}, frequency_range: [1, 45]}
parcellation:
  - parcellate:
      parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz
      orthogonalisation: symmetric
group:
  - sign_flip: {n_embeddings: 15, standardize: true}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=4, qc_dir=False)
