# A Python data reader for SPM format files

A simple(ish) data reader for SPM files. This is intended to provide basic spm-file data reading capability into OSL. The data are loaded into an SPM-like python object HOWEVER this is only intended to all the reading of already-preprocessed data, NOT to facilitate mixing preprocessing between SPM and python.

For example, an intended useage would be to complete a whole preprocessing pipeline in SPM before loading the data into python to run a GLM using general purpose tools. A not intended purpose would be to run part of a preprocessing pipeline in SPM before loading data into Python for source reconstruction. These MEG-specific analyses use lots of meta-data which are often assumed to be in a specific format. We cannot guarantee compatibility between how sensor locations are represented in SPM compared to MNE-Python. (However the reverse is more likely to be true - MNE-Python can save fif files which are easily readable by SPM.)


## Example usage

Files can be loaded through the `SPMMEEG` object.

```
D = D = osl.utils.spmio.SPMMEEG('/path/to/my/spmfile.mat')
```

A summary of the file contents can then be printed to the screen.

```
D.print_info()
```

producing the output....

```
SPM M/EEG data object - Loaded by OSL
Type: continuous
Transform: {'ID': 'time'}
1 conditions
388 channels
270400 samples/trial
1 trials
Sampling frequency 400Hz
Loaded from : /Users/andrew/Projects/ntad/analysis/ox_processed/dmmn_bl_raw_tsss.mat

Montages available : 1
	0 : AFRICA denoised data
Use syntax 'X = D.get_data(montage_index)[channels, samples, trials]' to get data
```

The data is loaded into a memory mapped array and can be accessed using the `get_data` method.

```
X = D.get_data()
```

A montage can be applied to the data by specifying the corresponding montage index in `get_data`. In this case only 1 montage is available - an AFRICA denoised montaged with index zero.

```
X_ica = D.get_data(0)
```

Meta data can be accessed using the helper methods similar to those on the matlab SPM object.

```
D.nsamples
D.nchannels
D.ntrials

D.condlist

planar_inds = D.indchantype('MEGPLANAR')
```

## Sources
This was compiled from previous scripts written by Evan Roberts (https://github.com/evanr70/py_spm) and Mark Hymers.
