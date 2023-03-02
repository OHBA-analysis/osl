#!/usr/bin/env python

"""
Run group analysis on data on the Wakeman-Henson dataset.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>

import os.path as op
import h5py

import numpy as np
from anamnesis import obj_from_hdf5file
import glmtools as glm

subjects_dir = "/ohba/pi/mwoolrich/datasets/WakemanHenson/ds117"

subjects_dir = "/Users/woolrich/homedir/vols_data/WakeHen"

nsubjects = 19
nsessions = 6
subjects_to_do = np.arange(0, nsubjects)
sessions_to_do = np.arange(0, nsessions)
subj_sess_2exclude = np.zeros([nsubjects, nsessions]).astype(bool)

first_level_contrasts = [15]
baseline_correct = True
rectify = True

# -------------------------------------------------------------
# %% Setup file names

glm_model_files = []
glm_time_files = []

glm_dir = op.join(subjects_dir, "glm")

design_matrix = []
subj_indices = []

for sub in subjects_to_do:
    for ses in sessions_to_do:
        if not subj_sess_2exclude[sub, ses]:

            sub_name = "sub" + ("{}".format(subjects_to_do[sub] + 1)).zfill(3)
            ses_name = "run_" + ("{}".format(sessions_to_do[ses] + 1)).zfill(2)
            subject = sub_name + "_" + ses_name

            glm_model_file = op.join(
                glm_dir, subject, "first_level_glm_model.hdf5"
            )
            glm_time_file = op.join(
                glm_dir, subject, "first_level_glm_model_times.npy"
            )

            if op.exists(glm_model_file) and op.exists(glm_time_file):
                glm_model_files.append(glm_model_file)
                glm_time_files.append(glm_time_file)

                # store which subject this belongs to
                subj_indices.append(sub)

# -------------------
# Setup group-level design matrix in GLM tools
print("\nSetting up group design matrix")

design_matrix = np.zeros([len(subj_indices), len(set(subj_indices))])
regressor_names = []

for subj_ind in set(subj_indices):
    regressor_names.append("Subject {}".format(subj_ind))
    design_matrix[np.where(np.array(subj_indices)==subj_ind)[0], subj_ind] = 1

des = glm.design.GLMDesign.initialise_from_matrices(
    design_matrix,
    regressor_names=regressor_names,
    contrasts = np.ones([1,len(set(subj_indices))]),
    contrast_names = ["Mean"]
)

des.plot_summary()

for first_level_contrast in first_level_contrasts:

    group_glm_model_file = op.join(
                glm_dir, "group_level_glm_model_con{}.hdf5".format(first_level_contrast)
            )
    group_glm_time_file = op.join(
                glm_dir, "group_level_glm_model_times_con{}.npy".format(first_level_contrast)
            )
    # Gather first level copes as data for group glm
    data = []
    for glm_time_file, glm_model_file in zip(glm_time_files, glm_model_files):

        # Load GLM
        print("Loading GLM:", glm_model_file)
        model = obj_from_hdf5file(glm_model_file, "model")
        epochs_times = np.load(glm_time_file)

        cope = model.copes[first_level_contrast, :, :]

        if rectify:
            cope = abs(cope)

        if baseline_correct:
            baseline_mean = np.mean(
                cope[:, epochs_times < 0],
                axis=1,
            )
            cope = cope - np.reshape(baseline_mean, [-1, 1])

        data.append(cope)

    data = np.asarray(data) # (subjects x spatial_channels x tpts_within_trial)

    # Create GLM data
    data = glm.data.TrialGLMData(data=data)

    # ------------------------------------------------------

    # Fit Model
    print("Fitting GLM")
    model = glm.fit.OLSModel(des, data)

    # Save GLM
    print("Saving GLM:", group_glm_model_file)
    out = h5py.File(group_glm_model_file, "w")
    #des.to_hdf5(out.create_group("design"))
    data.to_hdf5(out.create_group("data"))
    model.to_hdf5(out.create_group("model"))
    out.close()

    np.save(group_glm_time_file, epochs_times)



