#!/usr/bin/env python

"""
Run group analysis on data on the Wakeman-Henson dataset.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from anamnesis import obj_from_hdf5file
import nibabel as nib
from osl.source_recon import parcellation, rhino
from nilearn import plotting

subjects_dir = "/ohba/pi/mwoolrich/datasets/WakemanHenson/ds117"
subjects_dir = "/Users/woolrich/homedir/vols_data/WakeHen"

parcellation_file = "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_4d.nii.gz"
parcellation_file = "HarvOxf-sub-Schaefer100-combined-2mm_4d.nii.gz"

mask_file = "MNI152_T1_2mm_brain.nii.gz"

first_level_contrast = 15

glm_dir = op.join(subjects_dir, "glm")

# ------------------------------------------------------
# Load group GLM fit

group_glm_model_file = op.join(
            glm_dir, "group_level_glm_model_con{}.hdf5".format(first_level_contrast)
        )
group_glm_time_file = op.join(
    glm_dir, "group_level_glm_model_times_con{}.npy".format(first_level_contrast)
)

print("Loading GLM:", group_glm_model_file)
model = obj_from_hdf5file(group_glm_model_file, "model")
epochs_times = np.load(group_glm_time_file)

# ------------------------------------------------------
# Output parcel-wise stats as 3D nii file in MNI space at time point of interest

group_contrast = 0
stats_dir = op.join(subjects_dir, "glm_stats")

if not op.isdir(stats_dir):
    os.makedirs(stats_dir, exist_ok=True)

# output nii nearest to this time point in msecs:
tpt = 0.110 + 0.034
volume_num = np.abs(epochs_times-tpt).argmin() # nearest epoch time to tpt

cope_map = model.copes[group_contrast, :, volume_num]
nii = parcellation.convert2niftii(cope_map, parcellation.find_file(parcellation_file), parcellation.find_file(mask_file))

cope_fname = op.join(
    stats_dir,
    "cope_gc{}_flc{}_vol{}".format(group_contrast, first_level_contrast, volume_num),
)

# Save cope as nii file and view in fsleyes
print(f"Saving {cope_fname}")
nib.save(nii, cope_fname + '.nii.gz')
rhino.fsleyes([parcellation.find_file(mask_file), cope_fname + '.nii.gz'])

# plot png of cope on cortical surface
plotting.plot_img_on_surf(
    nii,
    views=["lateral", "medial"],
    hemispheres=["left", "right"],
    colorbar=True,
    output_file=cope_fname,
)
os.system('open {}'.format(cope_fname + '.png'))

# ------------------------------------------------------
# Plot time course of group stats for a specified parcel

parcel_ind = 5
print("Plotting COPE time course for parcel:", parcel_ind)

cope_timeseries = model.copes[group_contrast, parcel_ind, :]

plt.figure()
plt.plot(epochs_times, cope_timeseries)
plt.title(
    "abs(cope) for contrast {}, for parcel={}".format(
        first_level_contrast, parcel_ind
    )
)
plt.xlabel("time (s)")
plt.ylabel("abs(cope)")
plt.show()

# ------------------------------------------------------------------
# plot subject-specific copes for the first-level contrast

print("Loading GLM:", group_glm_model_file)
first_level_data = obj_from_hdf5file(group_glm_model_file, "data").data  # nsess x nparcels x ntpts

plt.figure()
plt.plot(epochs_times, first_level_data[:, parcel_ind, :].T)
plt.plot(epochs_times, cope_timeseries, linewidth=2, color='k')
plt.title(
    "abs(cope) for contrast {}, for parcel={}".format(
        first_level_contrast, parcel_ind
    )
)
plt.xlabel("time (s)")
plt.ylabel("abs(cope)")
plt.show()

# ------------------------------------------------------------------
# Write stats as 4D niftii file in MNI space.
# Note, 4th dimension is timepoint within an epoch/trial

cope_map = model.copes[group_contrast, :, :].T
nii = parcellation.convert2niftii(cope_map, parcellation.find_file(parcellation_file), parcellation.find_file(mask_file))

cope_fname = op.join(
    stats_dir,
    "cope_gc{}_flc{}".format(group_contrast, first_level_contrast, volume_num),
)

# Save cope as nii file and view in fsleyes
print(f"Saving {cope_fname}")
nib.save(nii, cope_fname + '.nii.gz')
rhino.fsleyes([parcellation.find_file(mask_file), cope_fname + '.nii.gz'])

# From fsleyes drop down menus Select "View/Time series"
# To see time labelling in secs:
# - In the Time series panel, select Settings (the spanner icon)
# - In the Time series settings popup, select "Use Pix Dims"

