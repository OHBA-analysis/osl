# EEG example

Source reconstruction of EEG data with a structural MRI but no digitised
electrode positions: the electrodes of the montage are used as the
headshape, and their positions rescaled to fit the subject's head.

1. `1_preprocess.py`: filtering, downsampling and artefact removal.
2. `2_coregister.py`: surfaces and coregistration. First find the
   `rescale` factors that make the electrodes fit the scalp in the
   coregistration tab of the report (the step is cheap to rerun), then keep
   them for the forward model.
3. `3_source_reconstruct.py`: a three-layer forward model, beamforming and
   parcellation.
4. `4_sign_flip.py`: the dipole sign ambiguity across sessions.
