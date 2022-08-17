"""
OSL Preprocessing
==================
An example tutorial describing how to configure and run batch preprocessing in OSL.

"""

#%%
# We start by importing OSL and MNE-python

import osl
import mne

#%%
# Next we can simulate some MEG data for this example analysis

raw = osl.utils.simulate_raw_from_template(5000)
raw.plot(n_channels=30)


#%%
# This generates some fairly realisatic MEG data with prominant alpha oscillations.

raw.plot_psd()

#%%
# Make data with some problems...

flat_channels = [10]
bad_channels = [5, 200]
bad_segments = [(600, 750)]

raw = osl.utils.simulate_raw_from_template(5000,
                                           flat_channels=flat_channels,
                                           bad_channels=bad_channels,
                                           bad_segments=bad_segments)

raw.plot(n_channels=30)

#%%

bad_annotations, flat_channels = mne.preprocessing.annotate_flat(raw)

raw.set_annotations(bad_annotations)
raw.info['bads'].extend(flat_channels)

#%%

raw.plot(n_channels=30)

#%%

raw = osl.preprocessing.osl_wrappers.detect_badsegments(raw, segment_len=150)

raw.plot(n_channels=30)
