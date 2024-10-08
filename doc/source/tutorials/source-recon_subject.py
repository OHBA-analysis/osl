"""
Single Subject Source Reconstruction Tutorial
=============================================

In this tutorial, we will step through how to do source reconstruction (and parcellation) on a single session of data from a subject in the Wakeman-Henson task MEG dataset. This is a public dataset consisting of 19 healthy individuals who performed a simple visual perception task. See `Wakeman & Henson <https://www.nature.com/articles/sdata20151>`_ for more details.
 
The steps we will follow are:
 
1. Downloading the data from OSF
2. Setup file names
3. Compute surfaces, perform coregistration, and compute forward model using batching
4. Temporal Filtering
5. Compute beamformer weights
6. Apply beamformer weights
7. Parcellation
8. Epoching

Note that most of these steps can be carried out more simply over multiple subjects using batching. See the "Group Analysis of Source-space Data" tutorial for an example of this. Here we break the steps down and run them manually, to give you insight into how it all works.

To run this tutorial you will need to have OSL and FSL installed, with the appropriate paths specified in your environment. See the instructions on the repo/read the docs for how to install these packages.


1. Downloading the raw data from OSF
************************************

The public Wakeman-Henson dataset provides MaxFiltered data. Note that the full dataset is available on `OpenNeuro <https://openneuro.org/datasets/ds000117/versions/1.0.4>`_. Here, we will work with just a single subject from this dataset, which can be downloaded from the OSF project website. 

Let's download the data. Note, to download the dataset you need ``osfclient`` installed. This can be installed by excuting the following code in a jupyter notebook cell:

``!pip install osfclient``

We can now download the data for the single subject we will look at. Note that this will be placed in your current working directory.

"""


import os

def get_data(name):
    print('Data will be in directory {}'.format(os.getcwd()))
    """Download a dataset from OSF."""
    if os.path.exists(f"{name}"):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p zxb6c fetch SourceRecon/data/{name}.zip")
    os.system(f"unzip -o {name}.zip")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset
get_data("wake_hen")

#%%
# 2. Setup File Names
# *******************
# Let's first setup all the file names we will need, for an example single session from a single subject from the Wakeman-Henson dataset. 


import os.path as op
from pprint import pprint
from osl import utils

# setup dirs
data_dir = './wake_hen'
recon_dir = op.join(data_dir, "recon")
out_dir = op.join(data_dir, "recon", "glm")

# structurals
sub_name = "sub{sub_num}"
smri_files_path = op.join(data_dir, sub_name, "anatomy", "highres001.nii.gz")
smri_files = utils.Study(smri_files_path).get()

# fif files
subject = "{subject}"
preproc_fif_files_path = op.join(data_dir, subject + "_meg", subject + "_meg_preproc_raw.fif")
preproc_fif_files = utils.Study(preproc_fif_files_path)
subjects = preproc_fif_files.fields['subject']
preproc_fif_files = preproc_fif_files.get()

# setup output file names
sflip_parc_files=[]
for subject in subjects:
    sflip_parc_files.append(op.join(recon_dir, subject, "sflip_parc.npy"))

print('subjects:')
pprint(subjects)

print('Structural files:')
pprint(smri_files)

print('Preproc fif files:')
pprint(preproc_fif_files)

print('Sign flipped parcellated files:')
pprint(sflip_parc_files)

#%%
# 3. Compute Surfaces, Coregistration and Forward Modelling
# *********************************************************
#
# Here, we set the options in the dictionary ``config``, and use ``source_recon.run_src_batch``.
# See the tutorial on "Coregistration with RHINO" for more on how this works.
# 
# We do not use the nose and headshape points as these were not acquired for this dataset.
# 
# Setting ``gridstep: 10`` means that the data will be source reconstructed to each point on a regular 3D grid, with spacings of 10mm.


from osl import source_recon

config = """
    source_recon:
    - extract_fiducials_from_fif: {}
    - compute_surfaces:
        include_nose: false
    - coregister:
        use_nose: false
        use_headshape: false
    - forward_model:
        model: Single Layer
        gridstep: 10
"""
fsl_dir = '~/fsl'
source_recon.setup_fsl(fsl_dir)

source_recon.run_src_batch(
    config,
    src_dir=recon_dir,
    subjects=subjects,
    preproc_files=preproc_fif_files,
    smri_files=smri_files,
)

#%%
# 4. Temporal Filtering
# *********************
# 
# We temporally filter the data to focus on the oscillatory content that we are interest in. 
# 
# Here, we assume that we will be doing an evoked response (ERF) analysis on the epoched task data, and so we filter to the frequency range where the evoked response is typically contained, i.e. between 1 and 30 Hz.
# 


import mne

chantypes = ["grad"]

# Get and setup the data
data = mne.io.read_raw_fif(preproc_fif_files[0], preload=True)
data = data.pick(chantypes)

# Filter to the beta band
print("Temporal Filtering")
data = data.filter(
    l_freq=3,
    h_freq=20,
    method="iir",
    iir_params={"order": 5, "btype": "bandpass", "ftype": "butter"},
)
print("Completed")

#%%
# 5. Compute beamformer weights
# *****************************
#
# We now compute the beamformer weights (aka filters). These are computed using the (sensors x sensors) data covariance matrix estimated from the preprocessed and the temporally filtered MEG data (contained in *raw*), and the forward models (contained inside the ``subjects[0]`` inside the directory ``recon_dir``. 
# 
# Note that this automatically ignores any bad time segments when calculating the beamformer filters.
# 
# Here we source reconstructing using just the gradiometers.
# 
# The MEG data in the Wakeman and Henson dataset has been maxfiltered and so the maximum rank is ~64. We therefore slightly conservatively set the rank to be 55. This is used to regularise the estimate of the data covariance matrix.
# 
# More generally, a dipole is a 3D vector in space. Setting ``pick_ori="max-power-pre-weight-norm"`` means that we are computing a scalar beamformer, by projecting this 3D vector on the direction in which there is maximum power. 


from osl.source_recon import rhino, beamforming, parcellation
      
# Make LCMV beamformer filters
# Note that this will exclude any bad time segments when calculating the beamformer filters
filters = beamforming.make_lcmv(
    recon_dir,
    subjects[0],
    data,
    chantypes,
    pick_ori="max-power-pre-weight-norm",
    rank={"grad": 55},
)

#%%
# 6. Applying beamformer weights
# ******************************
#
# We now apply the beamformer filters to the data to project the data into source space.
# 
# Note that although the beamformer filters were calculated by ignoring any bad time segments, we apply the filters to all time points including the bad time segments. This will make it easier to do epoching later.


print("Applying beamformer spatial filters")

# stc is source space time series (in head/polhemus space).
stc = beamforming.apply_lcmv(data, filters)

# Convert from head/polhemus space to standard brain grid in MNI space
recon_timeseries_mni, reference_brain_fname, recon_coords_mni, _ = \
        beamforming.transform_recon_timeseries(recon_dir, 
                                                subjects[0], 
                                                recon_timeseries=stc.data, 
                                                reference_brain="mni")

print("Completed")
print("Dimensions of reconstructed timeseries in MNI space is (dipoles x all_tpts) = {}".format(recon_timeseries_mni.shape))

#%%
# 7. Parcellation
# ******************
# 
# At this point, the data has been source reconstructed to dipoles (in this case, a scalar value) at each point on a regular 3D grid, with spacings of 10mm. We could then analyse the data across all these dipoles.
# 
# An alternative, is to map the data onto a brain parcellation. This reduces the number of samples in the space from number of dipoles down to number of parcels. Using a parcellation helps to boost the signal to noise ratio, boost correspondance between subjects, reduce the severity of multiple comparison correction when doing any statistics, and aids anatomical interpretability.
# 
# The parcellation we use here is a combination of cortical regions from the Harvard Oxford atlas, and selected sub-cortical regions from the Schaefer 100 parcellation. 
# 
# Let's take a look at the positions of the centres of each parcel in the parcellation.


parcellation_fname = 'HarvOxf-sub-Schaefer100-combined-2mm_4d_ds8.nii.gz'

# plot centre of mass for each parcel
p = parcellation.plot_parcellation(parcellation_fname)

#%%
# Compute Parcel Time-courses
# ---------------------------
# 
# We use this parcellation to compute the parcel time courses using the parcellation and the dipole time courses. Note that the output parcel timepoints includes all time points, including any bad time segments.
# 
# Let's now parcellate the data to compute parcel time courses. This is done using the "spatial_basis" method, where the parcel time-course 
# first principal component from all voxels, weighted by the spatial map for the parcel (see `here <https://pubmed.ncbi.nlm.nih.gov/25862259/>`_).
#
#
# Apply parcellation to (voxels x all_tpts) data contained in recon_timeseries_mni.
# The resulting parcel_timeseries will be (parcels x all_tpts) in MNI space
# where all_tpts includes bad time segments

parcel_ts, _, _ = parcellation.parcellate_timeseries(
    parcellation_fname, 
    recon_timeseries_mni, 
    recon_coords_mni, 
    "spatial_basis", 
    recon_dir,
)


#%%
# We now put the parcel time courses into a new MNE raw object *parc_raw*. This will allow us to easily perform epoching using MNE.


# reload raw data to ensure that the stim channel is in there
raw = mne.io.read_raw_fif(preproc_fif_files[0])

parc_raw = parcellation.convert2mne_raw(parcel_ts, raw)

print("Dimensions of parc_raw are (nparcels x all_tpts) = {}".format(parc_raw.get_data().shape))

#%%
# 8. Epoching
# ***********
# 
# We can now perform epoching. Note that any epochs (aka trials) that contain any bad time segments will be rejected at this point.

from osl import preprocessing

dataset = preprocessing.read_dataset(preproc_fif_files[0])
epochs = mne.Epochs(
    parc_raw,
    dataset["events"],
    dataset["event_id"],
    tmin=-1,
    tmax=3,
    baseline=(None, 0),
)

print("Dimensions of epochs are (good_epochs x parcels x tpts_within_epoch) = {}".format(epochs.get_data().shape))

#%%
# We can now plot a simple evoked response for this session of data, by averaging over all epochs (aka trials), for a selected parcel


import matplotlib.pyplot as plt
import numpy as np

parcel_ind = 5
print("Plotting group COPE time course for parcel:", parcel_ind)

# average over trials/epochs
erf = np.mean(epochs.get_data()[:, parcel_ind, :], axis=0)

plt.figure()
plt.plot(epochs.times, erf)
plt.title("ERF, for parcel={}".format(parcel_ind))
plt.xlabel("time (s)")
plt.ylabel("ERF")













