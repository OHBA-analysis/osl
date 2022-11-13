API
================

This page is the reference for the functions in OSL.

Preprocessing Functions
***********************

.. autosummary::
   :toctree: stubs

    osl.preprocessing.run_proc_chain
    osl.preprocessing.run_proc_batch


MNE Wrappers
************

.. currentmodule:: osl.preprocessing.mne_wrappers

.. autosummary::
   :toctree: stubs

    
    run_mne_annotate_flat
    
    run_mne_annotate_muscle_zscore
    
    run_mne_anonymous
    
    run_mne_apply_baseline
    
    run_mne_apply_ica
    
    run_mne_compute_current_source_density
    
    run_mne_drop_bad
    
    run_mne_epochs
    
    run_mne_find_bad_channels_maxwell
    
    run_mne_find_events
    
    run_mne_ica_autoreject
    
    run_mne_ica_raw
    
    run_mne_maxwell_filter
    
    run_mne_notch_filter
    
    run_mne_pick
    
    run_mne_pick_channels
    
    run_mne_pick_types
    
    run_mne_resample
    
    run_mne_set_channel_types
    
    run_mne_tfr_morlet
    
    run_mne_tfr_multitaper
    
    run_mne_tfr_stockwell
    

OSL Wrappers
************

.. currentmodule:: osl.preprocessing.osl_wrappers

.. autosummary::
   :toctree: stubs

    
    detect_badchannels
    
    detect_badsegments
    
    detect_maxfilt_zeros
    
    exists
    
    run_osl_bad_channels
    
    run_osl_bad_segments
    
    run_osl_ica_manualreject
    