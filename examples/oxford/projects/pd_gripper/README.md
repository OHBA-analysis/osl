Example scripts to preprocess and source reconstruct task data
--------------------------------------------------------------

These scripts differ from int_ext in the order in which we epoch the data. In these example scripts we:

- Preprocess the continuous sensor data: `1_preprocess.py`.
- Perform manual ICA: `2_manual_ica.py`.
- Coregister: `3_coregister.py`.
- Source reconstruct the epoched data: `4_source_reconstruct.py`.
- Sign flip the epoched source data: `5_sign_flip.py`.
- Epoch the sign flipped parcellated data: `6_epoch.py`
