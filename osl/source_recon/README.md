# OSL Source Reconstruction

Tools for source reconstructing MEEG files.

## Usage

Coregistration:
```
from osl.source_recon import setup_fsl, rhino

# Setup FSL
source_recon.setup_fsl("/path/to/fsl")

# Input files: raw fifs, structural MRIs and preprocessed fifs
raw_files = ['raw1.fif', 'raw2.fif', ...]
smri_files = ['smri1.nii.gz', 'smri2.nii.gz', ...]
preproc_files = ['preproc_raw1.fif', 'preproc_raw2.fif', ...]

# Subject IDs (will be subdirectorires in coreg_dir)
subjects = ['sub-001', 'sub-002', ...]

# Coregistration
recon.rhino.run_coreg_batch(
    coreg_dir="/path/to/output/dir"
    subjects=subjects,
    raw_files=raw_files,
    preproc_files=preproc_files,
    smri_files=smri_files,
    model="Single Layer",
    include_nose=True,
    use_nose=True,
)
```

Source reconstruction:
```
import numpy as np
from mne.beamformer import apply_lcmv_raw
from osl import preprocessing
from osl.source_recon import beamforming, parcellation

# Channels to use
chantypes = ["meg"]
rank = {"meg": 60}

print("Channel types to use:", chantypes)
print("Channel types and ranks for source recon:", rank)

parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

for preproc_file, subject in zip(preproc_files, subjects):
    # Load preprocessed data
    preproc_data = preprocessing.import_data(preproc_file)
    preproc_data.pick(chantypes)

    # Bandpass filter
    preproc_data = preproc_data.filter(
        l_freq=1, h_freq=45, method="iir", iir_params={"order": 5, "ftype": "butter"}
    )

    # Beamforming
    filters = beamforming.make_lcmv(
        subjects_dir='/path/to/output/dir',
        subject=subject,
        dat=preproc_data,
        chantypes=chantypes,
        weight_norm="nai",
        rank=rank,
    )
    src_data = apply_lcmv_raw(preproc_data, filters)
    src_ts_mni, _, src_coords_mni, _ = beamforming.transform_recon_timeseries(
        subjects_dir='/path/to/output/dir',
        subject=subject,
        recon_timeseries=src_data.data,
    )

    # Parcellation
    p = parcellation.Parcellation(parcellation_file)
    p.parcellate(
        voxel_timeseries=src_ts_mni,
        voxel_coords=src_coords_mni,
        method="spatial_basis",
    )
    parcel_ts = p.parcel_timeseries["data"]

    # Save parcellated data
    np.save("/path/to/src/dir/" + subject + ".npy", parcel_ts)
```
