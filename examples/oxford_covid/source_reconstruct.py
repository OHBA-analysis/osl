"""Source reconstruction: beamforming and parcellation.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
from osl import rhino, parcellation, preprocessing
from mne.beamformer import apply_lcmv_raw

# Directories
PREPROC_DIR = "/ohba/pi/knobre/cgohil/covid/preproc"
SRC_DIR = "/ohba/pi/knobre/cgohil/covid/src"
COREG_DIR = SRC_DIR + "/coreg"
PREPROC_FILE = PREPROC_DIR + "/sub-{0}_task-restEO_preproc_raw.fif"

SUBJECTS = ["004", "005"]

# Files
preproc_files = []
for subject in SUBJECTS:
    preproc_files.append(PREPROC_FILE.format(subject))

# Channels to use
chantypes = ["meg"]
rank = {"meg": 60}

print("Channel types to use:", chantypes)
print("Channel types and ranks for source recon:", rank)

parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

for preproc_file, subject in zip(preproc_files, SUBJECTS):
    print("\nSubject", subject)
    print("--------" + "-" * len(subject) + "\n")

    # Load preprocessed data
    preproc_data = preprocessing.import_data(preproc_file)
    preproc_data.pick(chantypes)

    # Bandpass filter
    preproc_data = preproc_data.filter(
        l_freq=1, h_freq=45, method="iir", iir_params={"order": 5, "ftype": "butter"}
    )

    # Beamforming
    filters = rhino.make_lcmv(
        subjects_dir=COREG_DIR,
        subject=subject,
        dat=preproc_data,
        chantypes=chantypes,
        weight_norm="nai",
        rank=rank,
    )
    src_data = apply_lcmv_raw(preproc_data, filters)
    src_ts_mni, _, src_coords_mni, _ = rhino.transform_recon_timeseries(
        subjects_dir=COREG_DIR,
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
    np.save(SRC_DIR + "/" + subject + ".npy", parcel_ts)
