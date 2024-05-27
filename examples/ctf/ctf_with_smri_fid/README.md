# CTF Example

In this example we coregister using sMRI fiducials.

## Pipeline

In this example we:

- `1_preprocess.py`: Preprocess the sensor-level data.
- `2_coregister.py`: Coregister the MEG and sMRI data.
- `3_source_reconstruct.py`: Beamform the sensor-level data and parcellate to give us the source-level data.
- `4_sign_flip.py`: Align the sign of each parcel time course to a template subject from the normative model.

## Parallelisation

See [here](https://github.com/OHBA-analysis/osl/tree/main/examples/parallelisation) for how to parallelise these scripts.
