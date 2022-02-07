"""
OSL Preprocessing
==================
An example tutorial describing how to configure and run batch preprocessing in OSL. This tutorial uses the publicly available "mmfaces" data from Wakeman & Henson (2015), which can be downloaded [here](ftp://ftp.mrc-cbu.cam.ac.uk/personal/rik.henson/wakemandg_hensonrn/). It contains MEG (Elekta), EEG, functional and structural MRI data from a task where 19 participants saw famous, unfamiliar, and scrambled faces during multiple runs. Here, we will use a single MEG run of two selected participants (note that in the original publication, subjects 1,5 and 16 were excluded from analysis): sub002_run_01_sss_raw.fif and sub003_run_01_sss_raw.fif.

"""

#%%
# We start by importing OSL, MNE-python, and YAML

import osl
import mne
import yaml

#%%
# Next, we will set the data directory which contains the downloaded data, and specify the filenames. We will also set the output directory.
datadir = '~/WakeHen/'
outdir = datadir + 'processed/'

filenames = ['sub002_run_01_sss_raw.fif', 'sub003_run_01_sss_raw.fif']
list_of_raw_files = [datadir + fname for fname in filenames]

# save the list of raw files as a text file
textfile = open("list_of_raw_files.txt", "w")
for element in list_of_raw_files:
    textfile.write(element + "\n")
textfile.close()

# Verify that the path to the data is correct:
print(list_of_raw_files)

#%%
# We will first load a single dataset and visualize the data
raw = mne.io.read_raw_fif(list_of_raw_files[0], preload=True)
raw.plot(n_channels=30)


#%% We will now set up the preprocessing pipeline. For this, we will create a configuration file with some metadata, and all the preprocessing steps we want to do (including the appropriate settings). The config is a dictionary of settings, saved as plain text.
#   In the "meta" part of the config, we will specify the trigger codes that belong to each condition. For example, in this dataset, the trigger code for the first time an examplar of a specific famous face is presented is 5. (Note: in case you didn't use any triggers (e.g. in resting state data), you can specify `event_codes: None`).
#   In the "preproc" part, we will specify the preprocessing options in the order we want them executed. OSL uses a combination of functions from MNE-python and OSL specific functions. Later in the tutorial, we will show how to use custom functions as well. If we look at each line within preproc, we see that they all start with "method". Most of these are equivalent to MNE-functions, e.g. "find_events" refers to mne.find_events, and "drop_bad" refers to mne.drop_bad. Those methods that don't directly correspond to an MNE-function will be OSL functions.

config_text= """
meta:
  event_codes:
    famous/first: 5
    famous/immediate: 6
    famous/last: 7
    unfamiliar/first: 13
    unfamiliar/immediate: 14
    unfamiliar/last: 15
    scrambled/first: 17
    scrambled/immediate: 18
    scrambled/last: 19
preproc:
  - crop:               {tmin: 30}
  - find_events:        {min_duration: 0.005}
  - set_channel_types:  {EEG061: eog, EEG062: eog, EEG063: ecg}
  - filter:             {l_freq: 0.1, h_freq: 175}
  - notch_filter:       {freqs: 50 100 150}
  - bad_channels:       {picks: 'mag'}
  - bad_channels:       {picks: 'grad'}
  - bad_channels:       {picks: 'eeg'}
  - bad_segments:       {segment_len: 2000, picks: 'mag'}
  - bad_segments:       {segment_len: 2000, picks: 'grad'}
  - bad_segments:       {segment_len: 2000, picks: 'eeg'}
  - interpolate_bads:   None
  - resample:           {sfreq: 400, npad: 'auto'}
  - ica_raw:            {n_components: 60, picks: 'meg'}
  - ica_autoreject:     {picks: 'meg', ecgmethod: 'correlation', apply: true}
  - epochs:             {tmin: -0.5, tmax: 1.5}
  - drop_bad:           {target: epochs, reject: {eog: 250e-6, mag: 4e-12, grad: 4000e-13}}
"""
#%% We can visualize our preprocessing pipeline as a flowchart:
osl.preprocessing.batch.plot_preproc_flowchart(config_text)

#%% We will now walk through the preprocessing options.
# We start by cropping the data, and finding events with a minimal segment length. We set the EEG channels that correspond to the two EOG channels and the ECG, and then filter all data.
# Next, for each data type we identify band channels and bad segments, and interpolate the bad segments. We downsample the data - note that the filter settings we used before have a low pass cut off at less then half the downsampling frequency, and we also adapted the notch filters to the intended downsampling frequency.
# We then run ICA on the MEG data. Note that `ica_raw` only runs the ICA algorithm, but doesn't include component rejection. We do this in `ica_autoreject`.
# Finally, we epoch the data around the events we defined earlier, and we remove bad epochs.

#%% Let's transform our config_text to YAML format. YAML is a data serialization format designed for human readability and interaction with scripting languages. The config is now a python dictionary which can be viewed and edited online.
config = yaml.load(config_text, Loader=yaml.FullLoader)

# view first stage
print(config['preproc'][0])

# Save the config so we can use it in the future
yaml.dump(config, open(outdir + 'config.yaml', 'w'))

#%% And now run the the preprocessing on a single dataset. We can either use the config we specified before, or load it from disk.
dataset = osl.preprocessing.run_proc_chain(raw, config)

# alternative:
config_from_file = osl.preprocessing.load_config(outdir + 'config.yaml')
dataset = osl.preprocessing.run_proc_chain(raw, config_from_file)

#%% `dataset` is a python dictionary that contains the preprocessed data and its derivatives: raw, ica, epochs, events, and event_id, which are all in the standard MNE-python format. Under the hood, all preprocessing calls in OSL, including those to MNE-python functions, have a dataset as input and as output argument.
dataset.keys()


#%% We can also preprocess multiple datasets in one go, and save them in outdir. Note that each MNE-python object (e.g. raw, epochs, events) will be saved in a seperate file with the default MNE extensions.
osl.preprocessing.run_proc_batch(config, list_of_raw_files, outdir, overwrite=True)

#%% Batch processing can also directly be done from the command line. See below for its usage (as run from he command line).
osl_batch --help

# For example, in order to execute the batch processing above from the command line, we call osl_batch with as first argument the configuration file, second argument the text file containing the paths to the raw data objects, third argument the output directory, and then any optional arguments, see below:
osl_batch my_config.yaml list_of_raw_files.txt ~/WakeHen/processed/ --overwrite

#%% So far we've only used MNE-python functions and OSL functions for preprocessing, but we might want to design custom functions for this. For illustration purposes, we will make a function which will reject IC components based on the kurtosis of the IC timecourse.
#   All preprocessing functions should be designed such that their first input and first output is a dataset dictionary. The second input argument should be `userargs`, which will contain all remaining input arguments. Inside the function, they can be retrieved using `userargs.get('parameter_name', defaultvalue)`.
#   Since OSL returns information on which processing stage is currently running, we might want to print out some useful information about our custom function.
#   The main body of the function deals with actually computing the kurtosis on the IC timecourses, and in turn, marks the components that surpass the threshold as bad, e.g. by adding their indices to dataset['ica'].exclude.

def ica_kurtosisreject(dataset, userargs):
    import numpy as np
    from scipy.stats import kurtosis
    threshold = userargs.get('threshold', 10)
    
    print('OSL Stage - {0}'.format('ICA Kurtosis Reject'))
    print('userargs: {0}'.format(str(userargs)))
    
    raw=dataset['raw']
    ica=dataset['ica']
    ic_map = ica.get_components()[mne.pick_types(ica.info, meg=True, eeg=False), :] # IC timecourses (sensors x number of ICs)
    ic_timeseries = np.transpose(
            np.matmul(np.transpose(ic_map), raw.get_data()[mne.pick_types(ica.info, meg=True, eeg=False), :])) # get the IC timeseries by multiplying the IC map with the Raw timeseries.
            
    k = kurtosis(ic_timeseries, fisher=False) # compute the kurtosis
    bad_components = np.where(k>threshold)[0] # find the components with a kurtosis value larger than the threshold
    if len(bad_components>0):
        dataset['ica'].exclude.extend(bad_components) # add these indices to ica.exclude.
    return dataset

#%% Let's replace the ica_autoreject in the config by ica_kurtosisreject
config['preproc'][14] = {'method': 'ica_kurtosisreject'}

# Now that our config contains a method that is not part of the OSL toolbox, we have to add the function as a user argument to `run_proc_chain` or `run_proc_batch`. This is done by specifying the argument `extra_funcs`. This is the command to run the preprocessing with the custom function on a single dataset:
dataset=osl.preprocessing.run_proc_chain(raw, config, extra_funcs=[ica_kurtosisreject])

# or on all datasets
osl.preprocessing.run_proc_batch(config, list_of_raw_files, outdir, overwrite=True, extra_funcs=[ica_kurtosisreject])



#%% The batch preprocessing is now complete. However, we might want to check the output data. It is good practice to inspect the data as quality control and make sure nothing unexpected happened. OSL has a handy tool for this, called the Report. The report will create a directory in a user specified location, in which a number of figures will be saved. It will also create an .html file, through which the report can be accessed.

# Generate the report on the preprocessed data
list_of_files = [outdir + fname for fname in filenames]
osl.report.gen_report(list_of_files, outdir=outdir+'report/')

# You can optionally run some preprocessing on each datafile before the figures are generated. This might be useful to remove known artefacts (eg trimming breaks at the start and end of scan or filtering out HPI signal) so that the report can focus on the MEG signal. The preprocessing is specified using the same config format as the osl preprocessing tool. For example, here we trim the first 20 seconds of data and apply a broad filter to the data before reporting.
config = """
meta:
  event_codes:
preproc:
  - crop:   {tmin: 20}
  - filter: {l_freq: 0.1, h_freq: 175}
"""

osl.report.gen_report(list_of_files, outdir='/path/to/save/dir', preproc_config=config)

#%% When we open the html file, we will see the names of the processed datasets on the top of the page. The "Preprocessing applied" below lists the extra preprocessing options from the config that we gave to osl.report.gen_report (i.e. it doesn't refer to the preprocessing we did at the top of this tutorial.
#   The report contains a panel for each dataset. Navigation through the different tabs can be done for all panels at once by using the tabs on top, or per dataset by using the tabs on the left side of the panel.

#%%   Below follows a specification of what can be found in each tab.
# - Info: Meta data including the path of the datafile, information about the recording, which data was collected, and the event codes that are present in the data
# - Time-Series: Shows a plot of a selection of channels, including information on bad segments. On the plot below you will find a time course of variance across sensors for each data type.
# - Channels: The distribution of per-sensor variance is plotted for the entire recording, and the channels that were annotated as bad are highlighted below.
# - Digitisation: Shows the digitisation of electrodes, HPI coils and Polhemus headshape points in 3D space
# - EOG: A time series of the EOG recoding, for both horizontal and vertical EOG.
# - ECG: Shows the electrocardiogram (top), the heart rate of the recording (bottom), and the temporally aligned ECG signatures (right).
# - ICA: Displays temporal, spectral and topographical information on each of the rejected ICA components
# - Power Spectra: Visualizes the power spectrum for each data type

