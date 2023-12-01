API
================

This page is the reference for the functions in OSL.


Maxfilter
*********

Wrappers for Elekta/MEGIN Maxfilter software (requires license).

.. currentmodule:: osl.maxfilter

.. autosummary::
   :toctree: stubs

    run_maxfilter

    run_multistage_maxfilter
    
    run_cbu_3stage_maxfilter

    run_maxfilter_batch
   


Preprocessing
*************


Pipeline Functions
++++++++++++++++++

Primary user-level functions for running an OSL pipeline.

.. currentmodule:: osl.preprocessing.batch

.. autosummary::
   :toctree: stubs
   
   run_proc_chain
   run_proc_batch


Utils
+++++

Utility functions for running an OSL pipeline.

.. currentmodule:: osl.preprocessing.batch

.. autosummary::
   :toctree: stubs
   
   append_preproc_info
   get_config_from_fif
   find_func
   import_data
   load_config
   plot_preproc_flowchart
   print_custom_func_info
   read_dataset
   write_dataset   


MNE Wrappers
++++++++++++

Wrappers for MNE functions to perform preprocessing.

.. currentmodule:: osl.preprocessing.mne_wrappers

.. autosummary::
   :toctree: stubs

    run_mne_anonymous
    
    run_mne_annotate_amplitude
    
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
++++++++++++

Wrappers for OSL functions to perform preprocessing.

.. currentmodule:: osl.preprocessing.osl_wrappers

.. autosummary::
   :toctree: stubs

    gesd

    detect_artefacts
    
    detect_badchannels
    
    detect_badsegments
    
    detect_maxfilt_zeros
    
    drop_bad_epochs
    
    exists
    
    run_osl_bad_channels
    
    run_osl_bad_segments
    
    run_osl_drop_bad_epochs
    
    run_osl_ica_manualreject


.. currentmodule:: osl.preprocessing.plot_ica

.. autosummary::
   :toctree: stubs

    plot_ica
    

Report
******
Sensor level
++++++++++++
.. currentmodule:: osl.report.raw_report

.. autosummary::
   :toctree: stubs

    gen_report_from_fif

   
Source level
++++++++++++
.. currentmodule:: osl.report.src_report

.. autosummary::
   :toctree: stubs
    
    gen_html_data

    gen_html_page

GLM
******



Source Reconstruction
**********************
Rhino
+++++

Parcellation
++++++++++++


Utilities
*********

File handling
+++++++++++++

.. currentmodule:: osl.utils.study

.. autosummary::
   :toctree: stubs
    
    Study

    Study.get


Logger
++++++++++++

.. currentmodule:: osl.utils.logger

.. autosummary::
   :toctree: stubs

    set_up

    set_level

    get_level

    log_or_print


Parallel processing
+++++++++++++++++++

.. currentmodule:: osl.utils.parallel

.. autosummary::
   :toctree: stubs

    dask_parallel_bag


Package Utilities
+++++++++++++++++

.. currentmodule:: osl.utils.package

.. autosummary::
   :toctree: stubs

    run_package_test

    soft_import


OPM Utilities
+++++++++++++

.. currentmodule:: osl.utils.opm

.. autosummary::
   :toctree: stubs

    convert_notts

    correct_mri