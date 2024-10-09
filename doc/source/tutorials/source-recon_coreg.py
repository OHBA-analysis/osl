"""
Coregistration
==============

What is coregistration??
************************
 
In MEG/EEG analysis we have a number of coordinate systems including:
 
* MEG (Device) space - defined with respect to  the MEG dewar.
* Polhemus (Head) space - defined with respect to the locations of the fiducial locations (LPA, RPA and Nasion). The fiducial locations in polhemus space are typically acquired prior to the MEG scan, using a polhemus device.
* sMRI (Native) space - defined with respect to the structural MRI scan.
* MNI space - defined with respect to the MNI standard space brain.
 
In order to compute foward models and carry out source reconstruction we need to have all of the necessary things (MEG sensors, dipoles, scalp surface etc.) in a common coordinate space. However, prior to coregistration, this is not the case. For example, we only have the MEG sensors located in MEG (device) space.
 
Coregistration is the process of learning a mapping between each pair of coordinate systems. To do this, we make use of landmarks (these act as a kind of Rosetta stone) whose locations are known in two different coordinate systems. Knowing where these landmarks are in two coordinate systems allows us to learn a mapping between the coordinate systems.
 
For example, the fiducials (LPA, RPA and Nasion) are known in both sMRI (Native) space and Polhemus (Head) space, and provide the information we need to learn a linear (affine) transform betwen sMRI (Native) space and Polhemus (Head) space.
 
The different landmarks, and the coordinate systems they are known in prior to coregistration, are summarised here:

Fiducial points
-----------------

A critical stage in the coregistration of polhemus (head) space to sMRI (Native) space is matching the fiducial points (LPA, RPA and nasion) in both spaces. It is therefore important to ensure that one gets as accurate fiducial locations as possible, particularly when using the polhemus system at the point of data acquisition.<br>

Headshape Points
----------------

It is nonetheless challenging to perfectly pinpoint the fiducial locations in both polhemus (head) space to sMRI (Native) space. It is for this reason that we also use headshape points to refine the coregistration. This proceeds as follows:

* learn an initial linear (affine) mapping from polhemus (head) space to sMRI (Native) space using just the fiducials (LPA, RPA and nasion)

* refine this linear (affine) mapping from polhemus (head) space to sMRI (Native) space using the headshape points derived from the polheums system and the scalp surface extracted from the structural MRI

Coregistration using RHINO
**************************

In OSL, the standard approach for coregistration is to use RHINO (Registration of Headshapes Including Nose in OSL). The RHINO pipeline is actually not just used for the coregistration, it can also:

* Compute the head and brain surfaces needed for coregistration and forward modelling from the strucural MRI (using FSL tools such as BET and Flirt)
* Perform coregistration so that the MEG sensors and head / brain surfaces can be placed into a common coordinate system.
* Compute the forward model (the lead fields) given the relative geometry of the MEG sensors and head / brain surfaces.

RHINO has a number of key elements that help it produce good coregistrations, including:

* Extraction of the full head surface including the nose. This is a key aspect of RHINO working well. It requires that we can locate the nose in both the structural MRI and in the Polhemus spaces. As such, we make the following recommendations when using RHINO:
    - Ensure a good quality structural MRI that allows extraction of the brain and scalp, and that the nose is included. The sMRI needs a field of view large enough to fully cover the head (NOTE: MRI operators not familiar with MEG often do not realise that the scalp and skull are important for MEG analysis and set the field of view to give maximum resolution of the brain by cropping the skull and scalp!).
    - Acquire a large number (>200) of Headshape Polhemus points. 
    - Acquire Headshape Polhemus points cover the scalp, brow and (rigid parts of) the nose. The surface matching algorithm searches for the best fit of the polhemus points to the surface extracted from the sMRI. Since the scalp is approximately spherical, an apparently good fit of the Polhemus points to the data can be achieved even if the fit is severely rotated away from the true position. This can be avoided by including the nose, which both constrains the fit and makes it easier to determine whether the coregistration has gone awry.
* Performing multi-start optimisation to avoid local minima. The surface matching algorithm is prone to local minima so it can often get stuck with a poor fit. This problem is greater if there are less points from which to estimate the fit. Any misleading/incorrect polhemus points should be removed.
* Extraction of the scalp surface directly from the structural MRI image using FSL tools to avoid distortion (this necessarily assumes the structural is distortionless).
    - Make sure the orientation information in the structural MRI niftii file is correct. Sometimes, the orientation information in the structural MRI nifti file is incorrect, for instance if it has been poorly converted from another format (e.g. DICOM). This can throw the RHINO fitting off, so review the anatomical markers (e.g. anterior/posterior) in FSLeyes and make sure that the sform is set correctly. A good sanity check is to open the structure in FSLeyes, alongside a standard space MNI brain, and make sure that the two images are roughly in the same location and have the same orientation (note - they do not need to be perfectly aligned, the surface extraction call below will do that registration for you).<br>


Running RHINO
=============

Here, we will demonstrate how to run RHINO on some example data from a CTF MEG scanner.
The steps we will follow in this tutorial are:

1. Downloading the data from OSF
2. Setup file paths
3. Compute surfaces
4. Coregistration
5. Compute forward model
6. Batched RHINO (combined surface extraction, coregistration and forward modelling over multiple subjects)

To run this tutorial you will need to have OSL and FSL installed, with the appropriate paths specified in your environment. See the instructions on the repo/read the docs for how to install these packages.


1. Downloading the raw data from OSF
*********************************

Let's download the two subjects' data from the OSF project website.

:note: To download the dataset you need ``osfclient`` installed. This can be installed by excuting the following code in a jupyter notebook cell:



"""


import sys
sys.command('pip install osfclient')


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
get_data("notts_2subjects")

#%%
# 2. Setup file paths
# ****************
# 
# Let's find the filepaths of the structural MRIs and preprocessed fif files on your computer.

import os
import os.path as op
from pprint import pprint
from osl import utils
from osl import source_recon
import numpy as np

data_dir = './notts_2subjects'
recon_dir = './notts_2subjects/recon'


subject = '{subject}'
smri_files_path = op.join(data_dir, subject,  subject + '_T1w.nii.gz')
smri_files = utils.Study(smri_files_path).get()

fif_files_path = op.join(data_dir, subject, subject + '_task-resteyesopen_meg_preproc_raw.fif')    
fif_files = utils.Study(fif_files_path)
subjects = fif_files.fields['subject']
fif_files = fif_files.get()

print('Structural files:')
pprint(smri_files)

print('fif files:')
pprint(fif_files)

print('subjects:')
pprint(subjects)

#%%
# :note: Please change the following directory to wherever you installed fsl
fsl_dir = '~/fsl'
source_recon.setup_fsl(fsl_dir)

#%%
# 3. Compute Surfaces
# *******************
# 
# The first thing we need to do is to use the structural image to compute the head and brain surfaces in Native and MNI space using *rhino.compute_surfaces*. The head and brain surfaces will be used later  for carrying out the coregistration and forward modelling.
# This step can be batched over multiple sessions/subjects, as we will see later. For now, we will do this for just the first subject. Note that this can take a few minutes.
# 
# The inputs we need to provide are for the first subject are:
# 
# * *smri_file* - the full path to the structural MRI niftii file
# * *recon_dir* - the full path to the directory that will contain the subject directories that RHINO will output
# * *subject* - the name of the subject directory that RHINO will output
# * *include_nose* - a boolean flag indicating whether or not to extract a head surface from the structural MRI that includes the nose. It your structural MRI includes the nose AND you have acquired polhemus headshape points that include the nose, then it is recommend to set this flag to True
# 

source_recon.rhino.compute_surfaces(
    smri_files[0],
    recon_dir,
    subjects[0],
    include_nose=True,
)

#%%
# We can now view the result using *fsleyes*. Note that *fsleyes* can sometimes take a few moments to open.
# 
# CHECK: in fsleyes that:
# 
# * The surfaces have been extracted properly compared with the structural
# * The nose is included in the scalp surface, if that was requested with the *include_nose* option above
# 
# If there are problems, then check that you have a sufficiently high quality MRI and that the MRI file has the correct orientation information. See more on this in the *Coregistration using RHINO* section above.
# 


source_recon.rhino.surfaces_display(recon_dir, subjects[0])

#%%
# 4. Coregistration
# *****************
# 
# Polhemus headshape points
# -------------------------
# 
# Before we run the actual coregistration, we need to provide the coordinates for the nasion, LPA, RPA and the headshape points in Polhemus (Head) space. These should be provided in millimetres. Typically, the polhemus coordinates can be extracted from the MEG *fif* file.
# 
# However, in this practical the polhemus files have been provided in these locations:
# 
# * *./notts_2subjects/sub-not001/polhemus/polhemus_nasion.txt*
# * *./notts_2subjects/sub-not001/polhemus/polhemus_rpa.txt*
# * *./notts_2subjects/sub-not001/polhemus/polhemus_lpa.txt*
# * *./notts_2subjects/sub-not001/polhemus/polhemus_headshape.txt*
# 
# These are ASCII text files that contain space separated (3 x num_coordinates) coordinates (e.g. *polhemus_nasion.txt* contains one column of 3 values). RHINO is hard-wired to look for these files in the these locations (where *./notts_2subjects/recon* is the *recon_dir* specified above).
# 
# * *./notts_2subjects/recon/sub-not001/rhino/coreg/polhemus_nasion.txt*
# * *./notts_2subjects/recon/sub-not001/rhino/coreg/polhemus_rpa.txt*
# * *./notts_2subjects/recon/sub-not001/rhino/coreg/polhemus_lpa.txt*
# * *./notts_2subjects/recon/sub-not001/rhino/coreg/polhemus_headshape.txt*
# 
# To handle this, we will now define and run a function, *copy_polhemus_files*, that will put the polhemus files for each subject into these standard RHINO locations. This function will also be used later when we use batching over multiple subjects (note that this is why the function needs to have unused inputs *preproc_file, smri_file, logger*).
# 


def copy_polhemus_files(recon_dir, subject, preproc_file, smri_file, logger):
    polhemus_headshape = np.loadtxt(op.join(data_dir, subject, 'polhemus/polhemus_headshape.txt'))
    polhemus_nasion = np.loadtxt(op.join(data_dir, subject, 'polhemus/polhemus_nasion.txt'))
    polhemus_rpa = np.loadtxt(op.join(data_dir, subject, 'polhemus/polhemus_rpa.txt'))
    polhemus_lpa = np.loadtxt(op.join(data_dir, subject, 'polhemus/polhemus_lpa.txt'))
    
    #  Get coreg filenames
    filenames = source_recon.rhino.get_coreg_filenames(recon_dir, subject)

    # Save
    np.savetxt(filenames["polhemus_nasion_file"], polhemus_nasion)
    np.savetxt(filenames["polhemus_rpa_file"], polhemus_rpa)
    np.savetxt(filenames["polhemus_lpa_file"], polhemus_lpa)
    np.savetxt(filenames["polhemus_headshape_file"], polhemus_headshape)

copy_polhemus_files(recon_dir, subjects[0], [], [], [])

sub1_polhemus_nasion = op.join(recon_dir, subjects[0], 'rhino/coreg/polhemus_nasion.txt')
print('E.g., the coordinates for the nasion for subject {} in Polhemus space are \n'.format(subjects[0]))
os.system('more {}'.format(sub1_polhemus_nasion))

#%%
# We can now perform coregistration so that the MEG sensors and head / brain surfaces can be placed into a common coordinate system.
# 
# We do this by running *rhino.coreg* and passing in:
#
# * *fif_file* the full path to the MNE raw fif file.
# * *recon_dir* - the full path to the directory that contains the subject directories RHINO  outputs
# * *subject* - the name of the subject directories RHINO outputs to
# * *use_headshape* - a boolean flag indicating whether or not to use the headshape points to refine the coregistration.
# * *use_nose* - a boolean flag indicating whether or not to use the nose headshape points to refine the coregistration. Setting this to True requires that include_nose was set True in the call to *rhino.compute_surfaces*, and requires that the polhemus headshape points include the nose.


source_recon.rhino.coreg(
    fif_files[0],
    recon_dir,
    subjects[0],
    use_headshape=True,    
    use_nose=True,
)


#%%
# We can now view the result. Note that here we set ``display_outskin_with_nose=False``, which means the nose is not shown in the visualisation even though it is used in the coregistration. We do this because creating a mesh with the nose included is computationally intensive.
# 
# The coregistration result is shown in MEG (device) space (in mm).
# 
# * Grey disks - MEG sensors
# * Blue arrows - MEG sensor orientations
# * Yellow diamonds - MRI-derived fiducial locations
# * Pink spheres - Polhemus-derived fiducial locations
# * Green surface - Whole head scalp extraction
# * Red spheres - Polhemus-derived headshape points
# 
# A good coregistration shows:
# 
# * MRI fiducials (yellow diamonds) in appropriate positions on the scalp
# * Polhemus-derived fiducial locations (pink spheres) in appropriate positions on the scalp 
# * Good correspondence between the headshape points (red spheres) and the scalp
# * The scalp appropriately inside the sensors, and with a sensible orientation.
# 
# If you have a bad co-registration:
# 
# * Go back and check that the compute_surfaces has worked well using ``fsleyes`` (see above).
# * Check for misleading or erroneous headshape points (red spheres) and remove them. See the `Deleting Headshape Points <https://osl.readthedocs.io/en/latest/tutorials_build/source-recon_deleting-headshape-points.html>`_ tutorial for how to delete headshape points.
# 
# * Check that the settings for using the nose are compatible with the available MRI and headshape points
# * The subject in question may need to be omitted from the ensuing analysis.
# 



source_recon.rhino.coreg_display(
        recon_dir,
        subjects[0],
        display_outskin_with_nose=False,
        filename='./coreg_dispay.html',
)

print('You can also view coreg display by opening this file in a web browser: \n{}'.format(os.getcwd() + ('/coreg_dispay.html')))

#%%
# 5. Compute Forward Model
# ************************
# 
# We can now compute the forward model (the lead fields) given we can now put the MEG sensors and head / brain surfaces in the same coordinate system. We do this by running *rhino.forward_model*. Note that this is mostly just a wrapper call for a standard MNE function.
# Here we are modelling the brain/head using 'Single Layer', which corresponds to just modelling the inner skull surface, which is the standard thing to do in MEG forward modelling.
# Lead fields will be computed for a regularly space dipole grid, with a spacing given by the passed in argument *gridstep*. The dipole grid is confined to be inside the brain mask as computed by *rhino.compute_surfaces*.
# 


gridstep = 10
source_recon.rhino.forward_model(
    recon_dir,
    subjects[0],
    model="Single Layer",
    gridstep=gridstep,
)

#%%
# We can now view the result. Note that that the small black points inside the brain show the locations of the dipoles that the leadfields have been computed for.


source_recon.rhino.bem_display(
    recon_dir,
    subjects[0],
    display_outskin_with_nose=False,
    display_sensors=True,
    plot_type="surf",
    filename='./bem_dispay.html',
    
)

print('You can also view BEM display by opening this file in a web browser: \n{}'.format(os.getcwd() + ('/bem_dispay.html')))

#%%
# We now have a *forward_model_file* stored in the rhino directory for this subject.
#     
# This file contains the leadfields that map from source to sensor space, and which are used to do source reconstruction.
# 


from mne import read_forward_solution

# load forward solution
fwd_fname = source_recon.rhino.get_coreg_filenames(recon_dir, subjects[0])["forward_model_file"]
fwd = read_forward_solution(fwd_fname)
leadfield = fwd["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

#%%
# 6. Batched RHINO
# ****************
# 
# So far we have shown how we can compute surfaces, do coregistration and forward modelling step-by-step on a single subject at a time.
# Alternatively, we can do batching over multiple subjects. This is much easier to organise in a script and automatically produces an HTML report page.
# To do this, we need to pass the following to the batching function:
# 
# * *config* - the settings for each of the steps: *compute_surfaces, coregister, forward_model*
# * *src_dir* - the path to the recon directory where results of the coreg etc will be placed
# * *subjects* - list of the names of the subject directories for all subjects
# * *preproc_files* - list of MNE raw file for all subjects
# * *smri_files* - list of structural MRI niftii files for all subjects
# * *extra_funcs* - this is where we pass the function we wrote earlier *copy_polhemus_files* which will be run to put the polhemus files for each subject in the correct locations for RHINO



config = """
    source_recon:
        - copy_polhemus_files: {}
        - compute_surfaces:
            include_nose: true
        - coregister:
            use_nose: true
            use_headshape: true
        - forward_model:
            model: Single Layer
            gridstep: 10
    """

source_recon.run_src_batch(
    config,
    src_dir=recon_dir,
    subjects=subjects,
    preproc_files=fif_files,
    smri_files=smri_files,
    extra_funcs=[copy_polhemus_files],
)

#%%
# Viewing the batched results
# ---------------------------
# 
# As the last part of the command line output from running the batch indicates, the results can be viewed by opening the following file in a web browser:


print('View a summary report by opening the following file in a web browser:\n{}'.format(os.getcwd() + ('/notts_2subjects/recon/report/subject_report.html')))













