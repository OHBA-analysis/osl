"""Source reconstruction on FreeSurfer surfaces instead of RHINO.

recon-all runs once per subject (hours) and the surface forward model and
minimum norm replace the volumetric beamformer; the parcellation is a
FreeSurfer annotation. Needs FreeSurfer; set FREESURFER_HOME or call
osl_ephys.source_recon.setup_freesurfer first.
"""

from osl_ephys.pipeline import run_batch
from osl_ephys.source_recon import setup_freesurfer

setup_freesurfer("/Applications/freesurfer/7.4.1")

sessions = [
    {
        "id": "sub-001_task-rest",
        "subject": "sub-001",
        "raw_file": "raw/sub-001_task-rest.fif",
        "smri_file": "smri/sub-001_T1w.nii.gz",
    }
]

config = """
preproc:
  - filter: {l_freq: 1, h_freq: 45}
  - resample: {sfreq: 250}
coreg:
  - recon_all: {}
  - make_watershed_bem: {}
  - extract_fiducials_and_headshape: {}
  - coregister: {n_init: 20}
source_recon:
  - forward_model: {spacing: oct6, ico: 4}
  - minimum_norm: {chantypes: [mag, grad], rank: {meg: 60}}
parcellation:
  - parcellate: {source_method: minimum_norm, parcellation_file: aparc, method: pca_flip}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", surface_extraction_method="freesurfer")
