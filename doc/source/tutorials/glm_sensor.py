"""
GLM-Spectrum with MEG Data
==========================
 
In the previous tutorial we introduced the concepts behind spectrum estimation and how this can be extended with General Linear Modelling. Here, we apply these principles to real MEG datasets. 
 
We will analyse the power spectra from the Wakeman & Henson face processing dataset used in the preprocessing tutorials. Though this is a task dataset, we will treat it as resting state for the purpose of this analysis.
 
We start with some preparation, let's import our modules as make the default font size for figures a little larger.
"""

import numpy as np
from scipy import signal
import osl
import os
import sails
import glmtools as glm
import mne
from pprint import pprint

import matplotlib
font = {'size' : '18'}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

#%%
# Downloading and importing an example dataset
# ********************************************
# We're going to use two datasets in this practical. One preprocessed MEG dataset to illustrate first-level GLM-Spectrum estimation, and one folder of fitted first level GLM-Spectra to illustrate group-level GLM-Spectra.
# 
# These are both available to download from the OSL-Workshop OSF page. The following code cell will download and unzip these files if they don't already exist on your computer. We're assuming that the current working directory of this notebook is the correct place to download the data. If you've already downloaded and unzipped the data by hand, then this cell should just tell you that everythiing is in place!
# 
# You'll need ``osfclient`` to be installed in your python environment for this to work. This should be included as a dependency in the workshop environment


preprocessed_meg = 'sub-09_ses-meg_task-facerecognition_run-01_meg_preproc_raw.fif'

if not os.path.exists(preprocessed_meg):
    print('Preprocessed MEG data not found, downloading 160MB...')
    os.system(f"osf -p zxb6c fetch GeneralLinearModelling/{preprocessed_meg}")
else:
    print('Preprocessed data found')
    
fl_dir = 'osl-workshop_glm-spectrum_first-levels'
fl_base = os.path.join(fl_dir, '{subj}_ses-meg_task-facerecognition_{run}_meg_glm-spec.pkl')

if os.path.exists(fl_dir) and os.path.exists(fl_base.format(subj='sub-01', run='run-01')):
    first_levels = osl.utils.Study(fl_base)
    
    if len(first_levels.get()) == 96:
        print('All first levels found')
    else:
        print('First levels partially found... something odd has happened')
    
else:
    print('First-Level GLM-Spectra data not found, downloading 887.4MB...')
    name = 'osl-workshop_glm-spectrum_first-levels'
    os.system(f"osf -p zxb6c fetch GeneralLinearModelling/{name}.zip")
    print('...extracting...')
    os.system(f"unzip -o {name}.zip -d {name}")
    os.remove(f"{name}.zip")
    first_levels = osl.utils.Study(fl_base)

#%%
# ``preprocessed_meg`` is now a path to a ``fif`` file containing a processed dataset and ``first_levels`` is OSL study object which contains all first-level GLM-Spectra.
# 
# Let's get started!
#
# A single data recording : first-level analysis
# **********************************************
# Our first analysis in this tutorial will involve a single MEG recording. We'll fit a separate GLM-Spectrum to each channel of our dataset. Remembering that single GLM-Spectrum already fits a separate GLM for each frequency bin, we see that we can easily end up fitting several hundreds of GLMs on each single dataset.
# 
# The same theory and priniciples from the simulations apply here. We'll plot the standard Welch's perioidogram before briefly explore the effect of changing the sliding window segment length (`nperseg`) on our real dataset. We will then build a more complex model that quantifies trends over time and accounts for how EOG artefacts might appear in our recording.
# 
# We'll start by loading a single file and doing a little preparation. We're going to look at the first dataset from subject nine - but do come back and run through this section after selecting a different dataset to get an idea of between subject variability.


# Load in a single run from subject 9
raw = mne.io.read_raw_fif(preprocessed_meg, preload=True)

# Extract and filter the EOG - we'll use this later...
eogs = raw.copy().pick_types(meg=False, eog=True)
eogs = eogs.filter(l_freq=3, h_freq=20, picks='eog').get_data()

# Extract only the gradiometers
raw = raw.copy().pick_types(meg='grad')

# Remove headshape points - none were recorded on the back of the head and this distorts our topoplots
mon = raw.get_montage()
mon.dig = [dd for dd in mon.dig if dd['kind'] < 4]
raw = raw.set_montage(mon)

#%%
# Let's get started by computing the standard spectrum using Welch's method. The default parameters for ``osl.glm.glm_spectrum`` do this for us.
# 
# We'll truncate the spectrum to values between 1.5Hz and 95Hz to clip parts of the spectrum that have been affected by bandpass filtering during preprocessing




glmspec1 = osl.glm.glm_spectrum(raw, fmin=1.5, fmax=95)

#%%
# The GLM equivalent to this approach is to fit a model with a single constant regressor. Let's take a look at the design matrix to confim that the model is as expected



fig = glmspec1.design.plot_summary()

#%%
# Good - we have a single regressor and a single contrast.
# 
# Next, we can visualise our spectrum. ``OSL`` help us out with this. If we pass in a ``Raw`` object to ``osl.glm.glm_spectrum`` then the output GLM-Spectrum retains some information about the sensors and data structure. In particular, the GLM-Spectrum output contains a copy of the ``Raw.info`` configuration.




glmspec1.info

#%%
# This output object can use the sensor information to build intuitive plots of GLM-Spectra computed on sensorspacee datasets. For example, we can plot the power spectrum using ``MNE``'s spatial colour scheme to label sensor positions.




plt.figure(figsize=(9,6))
ax = plt.subplot(111)
glmspec1.plot_sensor_spectrum(0, ax=ax, sensor_proj=True)

#%%
# Linear spacings on the x-axis of power spectra can squash the key information into the left hand side of the figure. A square-root frequency axis can decompress things a little by making the series of squares (1, 4, 9, 16, ...) equally spaced on the x-axis. We can adjust our plot to this scaling by specifying ``base=0.5``. 


plt.figure(figsize=(9,6))
ax = plt.subplot(111)
glmspec1.plot_sensor_spectrum(0, ax=ax, sensor_proj=True, base=0.5)

#%%
# Finally, we can make a ``joint`` plot which includes both the spectrum and a series of topoplots displaying the spatial topography at key frequencies in the spectrum. This is computed using ``plot_joint_spectrum``. Let's visualise the spatial maps at the two prominant peaks around 10.5Hz and 15Hz.



plt.figure(figsize=(9,6))
ax = plt.subplot(111)
glmspec1.plot_joint_spectrum(freqs=[10.5, 15], base=0.5, ax=ax)

#%%
# As shown in the simulations, the critical choice for determining the resolution of a sliding window based spectrum is the sliding window length set by ``nperseg``. By default, ``glm_spectrum`` will set the sliding window length to be equivalent to the data sampling rate if a ``Raw`` object is passed as the input - though we can override this default by specifying our own value for ``nperseg``.
# 
# Let's compute Welch's periodogram using three different sliding window lengths to see its effect.




plt.figure(figsize=(18,6))
plt.subplots_adjust(wspace=0.3)

sample_rate = raw.info['sfreq']
npersegs = np.array([sample_rate//2, sample_rate, sample_rate*4], dtype=int)

for ii in range(3):
    plt.figure(figsize=(9,6))
    ax = plt.subplot(111)
    glmspec1 = osl.glm.glm_spectrum(raw, fmin=1.5, fmax=95, nperseg=npersegs[ii])
    glmspec1.plot_joint_spectrum(contrast=0, freqs=[3, 10, 15], ax=ax)
    ax.set_title('Window Length : {}'.format(npersegs[ii]))

#%%
# As we saw with the simulations, longer window lengths give higher resolution spectrum estimates containing more frequency bins per Hz.
# 
# The default of setting the sliding window length to the sample rate provides a sensible starting point. This will give 1 frequency bin per Hz.
#
# GLM-Spectrum estimation
# ***********************
# So far in this tutorial, we've looked at the 'standard' Welch's periodogram method for computing spectra. Next, we're going to explore the utility of the GLM-Spectrum on real data.
# 
# Let's start by defining three additional regressors to add into our model. One zero-mean covariate that quantifies a linear trend in time and two non-zero-mean confound regressors that quantify the effect of the EOG channel and the bad segments identified in the dataset.
# 
# Our regressors will be processed differently. The zero-mean covariate will be passed into the keyword argument `reg_ztrans` to be z-transformed, whilst the confound regressors are passed into `reg_unitmax` to be scaled between zero and one.




# Compute a time-series indicating where bad segments appear in the data
bads = np.zeros((raw.n_times,))
for an in raw.annotations:
    if an['description'].startswith('bad'):
        start = raw.time_as_index(an['onset'])[0] - raw.first_samp
        duration = int(an['duration'] * raw.info['sfreq'])
        bads[start:start+duration] = 1

# Define dictionaries containing the covariate terms
covs = {'Linear': np.linspace(0, 1, raw.times.shape[0])}
cons = {'EOG': np.abs(eogs)[1, :],
        'BadSegs': bads}

# Compute the GLM-Spectrum
glmspec2 = osl.glm.glm_spectrum(raw, fmin=1.5, fmax=95, reg_ztrans=covs, reg_unitmax=cons)


# First, let's check the design matrix of our new model.



fig = glmspec2.design.plot_summary()

#%%
# Our four regressors are present as expected. Note that only the 'Linear' regressor contains any negative values.
# 
# We should also check the y-axis labels here to see the number of observations going into our model fit. This corresponds to the number of sliding window data segments computed in our Short-Time Fourier Transform. In this case, we have quite a long recording with over 900 sliding window segments to use to fit our model. This is plenty for four regressors.
# 
# Let's take a look at the GLM cope-spectra for each of the four contrasts. We'll use ``plot_joint_spectrum`` to visualise both the spectrum and the topography of the spectrum at a few key frequencies.



plt.figure(figsize=(18,15))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

for ii in range(4):
    ax = plt.subplot(2, 2, ii+1)
    glmspec2.plot_joint_spectrum(contrast=ii, freqs=[3, 10, 15], ax=ax, base=0.5)
    plt.title(glmspec2.design.contrast_names[ii])

#%%
# Quite a bit to unpack from this plot! Let's break it down.
# 
#  - The ``'Mean'`` spectrum corresponds to our constant regressor (though this actually computes an intercept as we've included other non-zero regressors in the model - see theory and simulation tutorial for more details). this contains the familiar 1/f-type power spectrum shape with a prominant alpha and beta peak around 10 and 15Hz respectively.
#  
#  - The ``'Linear'`` regressor has a sharp negative deflection around 9Hz which appears centered around occipital cortex. This suggests that occipital alpha power is decreasing over time during the data recording.
#  
#  - The ``'EOG'`` confound regressor has largest values at low frequencies around bilateral frontal sensors close to the eyes. Though we have done ICA cleaning on this data, we can see that some residual variabilty in low frequencies can still be associated with eye movements.
#  
#  - The ``'BadSegs'`` regressor also has its largest values around low frequencies but has a less structured topography than the EOGs.
#  
# The GLM-Spectrum has quantified all these effects in one shot across all sensors and all frequencies. Critically, this is a multiple regression so the parameter esimates for each regressor are partialled from the other regressors and only quantify the unique contribution of that regressor in describing the data.
# 
# We can explore the relationship between our regressors by exploring the 'efficiency' of the design matrix.
#  




fig = glmspec2.design.plot_efficiency()

#%%
# The three subplots of the design efficiency show
# 
# - **The correlation matrix** of the model regressors (with the constant term blanked out as it has no variance). There is some feint structure here but no large correlations.
# 
# - **The Singular-Value spectrum** of the design matrix (Singular-values are computed by the singular value decomposition and are conceptually similar to eigenvalues in PCA). The profile of singular values indicates how close our design is to being 'low-rank'. A perfectly orthogonal and efficient design will have a flat set of singular values all around 1, where as a low-rank model will have some singular values very close or equal to zero. A low-rank model indicates that certain combinations of the regression parameters will be hard to estimate.  While the statistics we use will account for this by inflating the relevant variances, it is good to be aware when this is happening. For example, this could help stop us from being misled by the results of any affected statistical tests, and in some cases may help motivate a change in the design.
# 
# - **The Variance-Inflation Factor** describes the extent to which each regressor can be predicted by the other regressors in the model. Typically, values above 5 (or sometimes 2 if you're being cautious) are taken as an indiction that a regressor might be co-linear with something else in the design matrix.
# 
# Together these factors indicate how 'efficient' our design is - in other words, how well we're going to be able to estimate the parameters of the model. The singular-value spectrum indicates this most closely. If this contains any zeros then there is no matrix inverse of our design, which means our parameter estimates will be 'minimum-norm' estimates. This can be ok, in some cases we might accept inflated variance around our estimates if there is good reason to keep the design matrix as it is. In other case, this is an indication to changes something in the design.
# 
# If you do want to change the design, the correlation matrix and VIF scores indicate where any co-linearity in the model is likely to be. If there are large correlations and VIF values for a particular set of regressors, you may want to consider merging them or removing some of them.
# 
# In this case, we have a pretty well formed design matrix and are happy to continue!
# 
# 
# So far, we've only looked at the point estimates of our GLM-Spectra - but we will often want to view the spectrum of t-values for each contrast as well. This accounts for the standard error around the estimate of each parameter. If we have a large parameter estimate that also has a large uncertainty around its value, then this will be reflected in its low t-statistic. In contrast, we may have a small parameter estimate that the model is confident about - this will have a high t-statistic even though the parameter estimate is small.
# 
# Let's take a look at the t-value spectra for our four contrasts.



plt.figure(figsize=(18,15))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

for ii in range(4):
    ax = plt.subplot(2, 2, ii+1)
    glmspec2.plot_joint_spectrum(contrast=ii, freqs=[3, 10, 15], ax=ax, base=0.5, metric='tstats')
    plt.title(glmspec2.design.contrast_names[ii])

#%%
# This is quite different to the point estimates in the cope-spectra!
# 
# The mean t-spectrum is now a very strange shape with much of the structure flattened. This is as mean and intercept terms of spectrum estimates will have all positive values. The t-test quantifing whether the point estimate is different form zero is then not very informative..
# 
# Much of the structure in the other three regressors has also flattened a bit. In particular, the large alpha effect in the 'Linear' regressor has pretty much gone - indicating that this estimate had a large standard error.
# 
# The strong frontal effect the low-frequencies of the EOG regressor remains, indicating that this effect is very strong.
#
# Group analysis - combining multiple datasets
# ********************************************
# 
# It is rarely the case that we want to study a single data recording. Though this can be interesting in and of itself, we typically want to combine results across multiple (or many!) recordings to try and identify if there are consistent effects that might generalise to a wider population.
# 
# A group analysis serves this purpose. In the context of General Linear Modelling, a group analysis takes in the results from 'first-level' analyses of single datasets and combines them with a 'second-level' or 'group-level' analysis. In the case of our GLM-Spectra, our second level dataset is the set of parameter estimates fitted across all individual recordngs. We describe variability across partcipants in this group dataset with a group design matrix to provide a final set of results.
# 
# As a example - let's say we fit first-level models with 4 regressors across 100 frequencies and 306 sensors. The copes for each first-level result would be a matrix of shape ``(4, 100, 306)``. If our second level analysis combines results across 48 particpants, the first level results would be combined into a group-dataset of shape ``(48, 4, 100, 306)``. A group design matrix might then contain 2 regressors describing between subject variability. So, the final fitted model will be a matrix of shape `(2, 4, 100, 306)` containing (2 group contrasts, 4 first level contrasts, 100 frequencies, 306 sensors).
# 
# As the second level is still just a GLM, the same principles about varability and t-statistics that we saw in the previous section still apply to group analyses.
# 
# Let's run a group analysis to illustrate these principles. We start by finding the our datafiles on disk. These files contain first-level GLM-Spectra that have already been fitted for you.



fl_dir = 'osl-workshop_glm-spectrum_first-levels'
fl_base = os.path.join(fl_dir, '{subj}_ses-meg_task-facerecognition_{run}_meg_glm-spec.pkl')

first_levels = osl.utils.Study(fl_base)

#%%
# We can visualise the individual results of the first run of the first 12 subjects. Note that the spectra are extremely variable between recordings. It can be hard to see whether there is anything consistent happening by eye. This is why we need the second group level model.


run = 'run-01'

contrast = 0  # The first contrast refers to the 'Constant' regressor

plt.figure(figsize=(18,18))
plt.subplots_adjust(hspace=0.5)
for ii in range(12):
    subj = 'sub-{}'.format(str(ii+1).zfill(2))
    fpath = first_levels.get(subj=subj, run=run)[0]
    
    glmsp = osl.glm.read_glm_spectrum(fpath)
    ax = plt.subplot(3, 4, ii+1)
    glmsp.plot_sensor_spectrum(contrast=contrast, base=0.5, ax=ax, title=subj)

#%%
# The overall GLM-Spectra are already variable across datasets, but remember that we've modelled 4 regressors the first level. We can also visualse these in the same way.
# 
# Our group model will describe how the first level COPEs vary over subjects, at, for example, each sensor and frequency. We're not just looking for consistency in the mean-spectrum, but can also look for group effects of the first-level effect of EOG. Let's take a look at the first level EOG spectra next.



run = 'run-01'

contrast = 2  # The third contrast refers to the 'EOG' regressor

plt.figure(figsize=(18,18))
plt.subplots_adjust(hspace=0.5)
for ii in range(12):
    subj = 'sub-{}'.format(str(ii+1).zfill(2))
    fpath = first_levels.get(subj=subj, run=run)[0]
    
    glmsp = osl.glm.read_glm_spectrum(fpath)
    ax = plt.subplot(3, 4, ii+1)
    glmsp.plot_sensor_spectrum(contrast=contrast, base=0.5, ax=ax, title=subj)

#%%
# Again, there is lot of variability, some runs show a strong positive effect in low frequencies in green/yellow channels (these correspond to frontal sensors). Potentially suggesting that some of the eye movemenet artefact has not been removed during preprocessing.
#
# Building a group model
# **********************
# 
# Next we're going to specify our group level design matrix. We have 96 datasets in this analysis with 6 data recordings from each of 16 participants. Let's create some vectors in a dictionary that specify which participant and run each recording belongs to..

group_info = {'subjs': np.repeat(np.arange(16), 6),
              'runs': np.tile(np.arange(6), 16)}


# Our first vector picks out the six runs of each of the 16 participants
print(group_info['subjs'])


# And the second vector picks out runs 1 to 6 from all participants
print(group_info['runs'])

#%%
# We'll use this information to create our design. First, we need to specify a config that outlines how the design matrix will be constructed.
# 
# We'll add a categorical regressor for each participant that models the mean across that participants six runs. A single contrast will combine across all 16 of these regressors to compute a group average. A final zero-mean parametric regressor will describe any effects that change linearly across the six recordings.


from glmtools.design import DesignConfig

DC = DesignConfig()

group_avg_contrast = {}
for ii in range(16):
    DC.add_regressor(name='Subj_{}'.format(ii), rtype='Categorical', datainfo='subjs', codes=ii)
    group_avg_contrast['Subj_{}'.format(ii)] = 1/16
DC.add_regressor(name='Run', rtype='Parametric', datainfo='runs', preproc='z')


DC.add_contrast(name='GroupAvg', values=group_avg_contrast)
DC.add_contrast(name='Run', values={'Run': 1})

#%%
# Now we can fit our model! 
# 
# We use ``osl.glm.group_glm_spectrum`` to compute a group model. This takes a list of first-level models (a list of either the models themselves or the file paths of pickle files containing the models) as the first argument. These models are loaded into memory and concatenated to create the group dataset.
# 
# We'll also pass in the design config and the group info, these variables will be combined to make the group design. Finally, the model is fitted and the result returned in an object.



glmsp = osl.glm.group_glm_spectrum(first_levels.get(), design_config=DC, datainfo=group_info)

#%%
# Let's explore that in more detail. First, we'll visualise the group design matrix (making some tweaks to the plotting as this is a big design matrix...)



figargs = {'figsize': (18, 9)}
with plt.rc_context({'font.size': 10}):
    fig = glmsp.design.plot_summary(figargs=figargs)

#%%
# The design matrix has 96 rows as expected, one for each dataset.
# 
# We see our 16 regressors quantifying the mean of the six runs for each particpant and the parametric regressor looking at differences in runs in the final column. The first contrast combines across the 16 mean terms to make a group average and the second contrast simply isolates the final 'run' regressor.
# 
# Next we can take a look at the group data.



print(glmsp.data.data.shape)

#%%
# This is our 4-dimensional group data. We have 96 datasets, 4 first-level contrasts, 204 channels and 101 frequencies in this dataset. 
# 
# We can do a quick check to see if any of the 96 datasets are an obvious outlier. The ``plot_outliers`` function computes the variability across the final three dimensions to visualise a vector with one number per dataset. We can see that the variability within each of the 96 datasets is pretty comparable across the group.




fig = glmsp.data.plot_outliers()


# So, we've seen the ingredients. Let's take a look at the group model.
# 
# The 4-dimensonal array of group results has the expected shape. This is the same as the group data, but the 96 datasets in the first dimension have been reduced to 2 group level contrasts.



print(glmsp.model.copes.shape)


# The group GLM-Spectra themselves can be visualised using similar figures the first levels. Here we use ``plot_joint_spectrum``` to visualise the group average (group contrast 0) of each of the first level contrasts in turn.


plt.figure(figsize=(12,12))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

for ii in range(4):
    ax = plt.subplot(2, 2, ii+1)
    ylabel = 'Magnitude' if ii == 0 else ''
    glmsp.plot_joint_spectrum(gcontrast=0, fcontrast=ii, freqs=[3, 10, 15], ax=ax, base=0.5, ylabel=ylabel)

#%%
# There is lots of structure in these group averages. We can see
# 
# - The overall average has a familiar 1/f-type slope interrupted by a prominant alpha peak around 9Hz, and a prominant beta peak around 15Hz. The beta peak is very strong as we're analysing a task dataset which includes a motor response. We would likly only see the alpha peak in a normal resting state dataset
# 
# - The linear trend response has a broad peak between 5 and 9Hz covering bilateral temporal sensors. This indicates that the low alpha/theta power in these sensors increased over time within each recording.
# 
# - The EOG spectrum has a very strong bilateral frontal increase indicating increases in power associated with increased eye movement.
# 
# - The bad segments show the strongest response at low frequencies across a range of channels.
# 
# A lot going on but these are still only the point estimates of each regressor. We need to look at the t-statistics to get an idea of of the size of any statistical effect.
# 
# Let's repeat our plot but using the t-stats rather than the copes.




plt.figure(figsize=(12,12))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

for ii in range(4):
    ax = plt.subplot(2, 2, ii+1)
    glmsp.plot_joint_spectrum(gcontrast=0, fcontrast=ii, 
                              freqs=[3, 10, 15], ax=ax, 
                              base=0.5, ylabel='t-stat',
                              metric='tstats')

#%%
# Again lots of structure, but now we can make a stronger interpretation of the units on the y-axis. These are t-statistics which show the magnitude of an effect as a ratio with its stanard error. t-stats close to zero indicate that any effect is insubstantial compared to its variance.
# 
# We see a broadly similar structure to the point estimates but now can see that the low frequency EOG effect is likely to be very strong. The linear trend and bad-segment effects are still substantial but have much smaller t-statistics.
# 
# Finally, we can also explore the extent to which our first level parameter estimates varied across the six runs of each participant. Let's repeat our plot one more time but selecting the second group level contrast.




plt.figure(figsize=(12,12))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

for ii in range(4):
    ax = plt.subplot(2, 2, ii+1)
    glmsp.plot_joint_spectrum(gcontrast=1, fcontrast=ii, 
                              freqs=[3, 10, 15], ax=ax, 
                              base=0.5, ylabel='t-stat',
                              metric='tstats')

#%%
# The t-values here are much smaller but there is some structure. Perhaps the area around alpha in the change over runs of the mean spectrum shows an effect.
# 
# To formally assess an effect we need more than the t-value. It may have been the case that we were just lucky to observe a given effect with our data - but how can we quantify whether an effect was likely to have occured by chance?
# 
# We assess this using non-parametric permutation statistics. This is a numerical method for quantifying how often a particular result could have occured according to a particular null hypothesis. This is a pragmatic approach which simulates models that remove structure in the design matrix in accordance with the null hypothesis. We can compute hundreds or thousands of these null models and place their statistical estimates into a 'null' distrbution. Our observed statistic can then be compared to this null to create an estimate of how likely our result could have happened by chance.
# 
# Let's take the group mean of the EOG effect as a specific example. Our null hypthesis is that the GLM-Spectra of the EOG effect is no different from zero. If this were true, we would expect the parameter estimates of the model to be randomly distributed around zero. In turn, if this is true - then flipping the sign of half our first-level parameter estimates will not change the group result.
# 
# So, to assess this, we'll compute a few hundred 'null' models in which we flip the sign of the group-level mean regressor. If the real group mean is not different from zero then its value should fall well within this null distribution. If our real result would be very unlikely to have occurred by chances, then it should fall on the tails of the null distribution.
# 
# Here, we show an example permuted design matrix for a single 'null' model.




perm_design = glm.permutations.permute_design_matrix(glmsp.design, np.arange(16), 'sign-flip')
figargs = {'figsize': (18, 9)}
with plt.rc_context({'font.size': 10}):
    fig = perm_design.plot_summary(figargs=figargs)

#%%
# Notice that half our mean regressor values have been flipped. Try running the previous cell multiple times to see how the permutations change.
# 
# Now we run our stats themselves. We'll create 250 null models and use cluster statistics to control for multiple comparisons. For each null we will
# 
# - Permute the design matrix
# - Re-fit the group model
# - Identify clusters across sensors and frequency
# - Take the largest cluster statistic and add it to the null distribution
# 
# Then, we assess significance by
# 
# - Computing the observed statistics
# - Finding clusters in the result
# - Comparing the observed cluster statisticis to the null distribution
# - Keeping clusters which fall beyond the 95th percentile of the null.
# 
# Let's run the permutations for our group average of the first level EOG effect




P = osl.glm.ClusterPermuteGLMSpectrum(glmsp, 0, 2, nperms=50, cluster_forming_threshold=9)


# Once the permutations are complete - we can visualise the significant clusters.
plt.figure(figsize=(9, 9))
ax = plt.subplot(111)
P.plot_sig_clusters([99], base=0.5, ax=ax)

#%%
# We find a single significant cluster at low frequencies around the frontal sensors.
#
# Further Reading
# ***************
#
#   Wakeman, D. G., & Henson, R. N. (2015). A multi-subject, multi-modal human neuroimaging dataset. In Scientific Data (Vol. 2, Issue 1). Springer Science and Business Media LLC. https://doi.org/10.1038/sdata.2015.1 
# 
#   Quinn, A. J., Atkinson, L., Gohil, C., Kohl, O., Pitt, J., Zich, C., Nobre, A. C., & Woolrich, M. W. (2022). The GLM-Spectrum: A multilevel framework for spectrum analysis with covariate and confound modelling. Cold Spring Harbor Laboratory. https://doi.org/10.1101/2022.11.14.516449 






