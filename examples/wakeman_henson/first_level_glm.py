#!/usr/bin/env python

"""
Run group analysis on data from the Wakeman-Henson dataset.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>

import os
import os.path as op

import numpy as np
import h5py

import mne
import glmtools as glm
import osl

subjects_dir = "/ohba/pi/mwoolrich/datasets/WakemanHenson/ds117"
subjects_dir = "/Users/woolrich/homedir/vols_data/WakeHen"

hilb_freq_range = None
if True:
    hilb_freq_range = (7, 13)

nsubjects = 19
nsessions = 6
subjects_to_do = np.arange(0, nsubjects)
sessions_to_do = np.arange(0, nsessions)
subj_sess_2exclude = np.zeros([nsubjects, nsessions]).astype(bool)

# -------------------------------------------------------------
# %% Setup file names

preproc_fif_files = []
input_fif_files = []
epoch_fif_files = []
abs_epoch_fif_files = []
glm_model_files = []
glm_time_files = []
subjects = []

recon_dir = op.join(subjects_dir, "recon")
glm_dir = op.join(subjects_dir, "glm")

if not os.path.isdir(glm_dir):
    os.makedirs(glm_dir)

for sub in subjects_to_do:
    for ses in sessions_to_do:
        if not subj_sess_2exclude[sub, ses]:

            sub_name = "sub" + ("{}".format(subjects_to_do[sub] + 1)).zfill(3)
            ses_name = "run_" + ("{}".format(sessions_to_do[ses] + 1)).zfill(2)
            subject = sub_name + "_" + ses_name

            # output files
            preproc_fif_file = op.join(
                subjects_dir, subject + "_meg", subject + "_meg_preproc_raw.fif"
            )

            input_fif_file = op.join(
                recon_dir, subject, "sflip_parc-raw.fif"
            )

            epoch_fif_file = op.join(
                recon_dir, subject, "epoch_sflip_parc-epo.fif"
            )

            abs_epoch_fif_file = op.join(
                recon_dir, subject, "abs_epoch_sflip_parc-epo.fif"
            )

            glm_model_file = op.join(
                glm_dir, subject, "first_level_glm_model.hdf5"
            )
            glm_time_file= op.join(
                glm_dir, subject, "first_level_glm_model_times.npy"
            )

            if op.exists(input_fif_file) and op.exists(preproc_fif_file):
                preproc_fif_files.append(preproc_fif_file)
                input_fif_files.append(input_fif_file)
                epoch_fif_files.append(epoch_fif_file)
                abs_epoch_fif_files.append(abs_epoch_fif_file)

                glm_model_files.append(glm_model_file)
                glm_time_files.append(glm_time_file)
                subjects.append(subject)

                glm_subj_dir = op.join(glm_dir, subject)
                if not os.path.isdir(glm_subj_dir):
                    os.makedirs(glm_subj_dir)


# -------------------
# Epoch first-level design matrix
for preproc_fif_file, input_fif_file, epoch_fif_file, abs_epoch_fif_file \
        in zip(preproc_fif_files, input_fif_files, epoch_fif_files, abs_epoch_fif_files):

    raw = mne.io.read_raw(input_fif_file) # e.g. sensor, source space, or parcellated data

    # Epoch
    dataset = osl.preprocessing.read_dataset(preproc_fif_file)
    epochs = mne.Epochs(
        raw,
        dataset["events"],
        dataset["event_id"],
        tmin=-1,
        tmax=2.5,
        baseline=(None, 0),
        reject_by_annotation=True,
    )

    epochs.drop_bad(verbose=True)
    epochs.load_data()
    epochs.save(epoch_fif_file, overwrite=True)

    #### hilb

    # Do hilbert transform
    hilb_raw = raw.copy()
    hilb_raw.load_data()

    if hilb_freq_range is not None:
        hilb_raw.filter(
            l_freq=hilb_freq_range[0],
            h_freq=hilb_freq_range[1],
            method="iir",
            picks='all',
            iir_params={"order": 5, "btype": "bandpass", "ftype": "butter"},
        )

    hilb_raw.load_data()
    hilb_raw.apply_hilbert(picks='all')

    # Epoch
    dataset = osl.preprocessing.read_dataset(preproc_fif_file)
    abs_epochs = mne.Epochs(
        hilb_raw,
        dataset["events"],
        dataset["event_id"],
        tmin=-1,
        tmax=3,
        baseline=(None, 0),
        reject_by_annotation=True,
    )

    abs_epochs.drop_bad(verbose=True)
    abs_epochs.load_data()
    abs_epochs.save(abs_epoch_fif_file, overwrite=True)

if False:
    # Use this to copy minimum epoched files necessary for workshop practical

    #workshop_subjects_dir = '/Users/woolrich/CloudDocs/workshop/coreg_clean/data/wake_hen_group'
    workshop_subjects_dir = '/Users/woolrich/workshop/coreg_clean/data/wake_hen_group'
    workshop_recon_dir = op.join(workshop_subjects_dir, 'recon')

    # copy subject epoch fif files
    for subject in subjects:

        os.system("mkdir {}".format(op.join(workshop_recon_dir, subject)))

        file_from = op.join(recon_dir, subject, "epoch_sflip_parc-epo.fif")
        file_to = op.join(workshop_recon_dir, subject + '/')
        os.system("cp -f {} {}".format(file_from, file_to))

        file_from = op.join(recon_dir, subject, "abs_epoch_sflip_parc-epo.fif")
        file_to = op.join(workshop_recon_dir, subject + '/')
        os.system("cp -f {} {}".format(file_from, file_to))

# -------------------
# Setup first-level design matrix
print("\nSetting up design matrix")

DC = glm.design.DesignConfig()
DC.add_regressor(name="FamousFirst", rtype="Categorical", codes=5)
DC.add_regressor(name="FamousImmediate", rtype="Categorical", codes=6)
DC.add_regressor(name="FamousLast", rtype="Categorical", codes=7)
DC.add_regressor(name="UnfamiliarFirst", rtype="Categorical", codes=13)
DC.add_regressor(name="UnfamiliarImmediate", rtype="Categorical", codes=14)
DC.add_regressor(name="UnfamiliarLast", rtype="Categorical", codes=15)
DC.add_regressor(name="ScrambledFirst", rtype="Categorical", codes=17)
DC.add_regressor(name="ScrambledImmediate", rtype="Categorical", codes=18)
DC.add_regressor(name="ScrambledLast", rtype="Categorical", codes=19)
DC.add_simple_contrasts()
DC.add_contrast(
    name="Famous", values={"FamousFirst": 1, "FamousImmediate": 1, "FamousLast": 1}
)
DC.add_contrast(
    name="Unfamiliar",
    values={"UnfamiliarFirst": 1, "UnfamiliarImmediate": 1, "UnfamiliarLast": 1},
)
DC.add_contrast(
    name="Scrambled",
    values={"ScrambledFirst": 1, "ScrambledImmediate": 1, "ScrambledLast": 1},
)
DC.add_contrast(
    name="FamScram",
    values={
        "FamousFirst": 1,
        "FamousLast": 1,
        "ScrambledFirst": -1,
        "ScrambledLast": -1,
    },
)
DC.add_contrast(
    name="FirstLast",
    values={
        "FamousFirst": 1,
        "FamousLast": -1,
        "ScrambledFirst": 1,
        "ScrambledLast": 1,
    },
)
DC.add_contrast(
    name="Interaction",
    values={
        "FamousFirst": 1,
        "FamousLast": -1,
        "ScrambledFirst": -1,
        "ScrambledLast": 1,
    },
)
DC.add_contrast(
    name="Visual",
    values={
        "FamousFirst": 1,
        "FamousImmediate": 1,
        "FamousLast": 1,
        "UnfamiliarFirst": 1,
        "UnfamiliarImmediate": 1,
        "UnfamiliarLast": 1,
        "ScrambledFirst": 1,
        "ScrambledImmediate": 1,
        "ScrambledLast": 1,
    },
)
print(DC.to_yaml())


# -------------
# Fit first-level GLM
use_hilbert = True

if use_hilbert:
    use_abs = True
    epoch_fif_files_to_use = abs_epoch_fif_files
else:
    use_abs = False
    epoch_fif_files_to_use = epoch_fif_files

for epoch_fif_file, glm_model_file, glm_time_file \
        in zip(epoch_fif_files_to_use, glm_model_files, glm_time_files):

    epochs = mne.read_epochs(epoch_fif_file) # e.g. sensor, source space, or parcellated data

    epochs.drop_bad(verbose=True)
    epochs.load_data()

    # Load data in glmtools
    data = glm.io.load_mne_epochs(epochs)

    if use_abs:
        data.data = np.abs(data.data)

    # Create Design Matrix
    des = DC.design_from_datainfo(data.info)

    # ------------------------------------------------------

    # Fit Model
    print("Fitting GLM")
    model = glm.fit.OLSModel(des, data)

    # Save GLM
    print("Saving GLM:", glm_model_file)
    out = h5py.File(glm_model_file, "w")
    des.to_hdf5(out.create_group("design"))
    data.to_hdf5(out.create_group("data"))
    model.to_hdf5(out.create_group("model"))
    out.close()

    np.save(glm_time_file, epochs.times)

