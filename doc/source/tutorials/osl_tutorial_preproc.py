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
