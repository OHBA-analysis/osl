# OPM Example

Preprocessing and source reconstruction of OPM data.

## Pipeline

- `1_preprocess.py`: Preprocesses the data. This includes filtering, downsampling and automated artefact removal.
- `2_coregister.py`: Extract surfaces from the structural MRI, create the coregistration files OSL is expecting, and calculate the forward model.
- `3_source_reconstruct.py`: Beamform and parcellate.
- `4_sign_flip.py`: Sign flipping to fix the dipole sign ambiguity.
- `5_save_npy.py`: Save parcel data as numpy files.
