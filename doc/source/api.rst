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
    

Utils
+++++

Utility functions for running an OSL Preprocessing pipeline.

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


Report
******
Sensor level
++++++++++++
.. currentmodule:: osl.report.raw_report

.. autosummary::
   :toctree: stubs

    gen_report_from_fif

    gen_html_data

    gen_html_page

    gen_html_summary

   
Source level
++++++++++++
.. currentmodule:: osl.report.src_report

.. autosummary::
   :toctree: stubs
    
    gen_html_data

    gen_html_page

    gen_html_summary



GLM
******

Modality specific wrappers for glmtools.

GLM Base
++++++++++++++++++++++++

.. currentmodule:: osl.glm.glm_base

.. autosummary::
   :toctree: stubs

   GLMBaseResult

   GLMBaseResult.save_pkl

   GroupGLMBaseResult

   GroupGLMBaseResult.get_channel_adjacency

   BaseSensorPerm

   BaseSensorPerm.save_pkl

   SensorMaxStatPerm

   SensorMaxStatPerm.get_sig_clusters

   SensorClusterPerm

   SensorClusterPerm.get_sig_clusters

GLM Epochs
++++++++++++++++++++++++

.. currentmodule:: osl.glm.glm_epochs

.. autosummary::
   :toctree: stubs

    glm_epochs

    group_glm_epochs

    read_mne_epochs

    read_glm_epochs

    GLMEpochsResult

    GLMEpochsResult.get_evoked_contrast

    GLMEpochsResult.plot_joint_contrast

    GroupGLMEpochs

    GroupGLMEpochs.get_evoked_contrast

    GroupGLMEpochs.plot_joint_contrast

    GroupGLMEpochs.get_channel_adjacency

    GroupGLMEpochs.get_fl_contrast



GLM Spectrum
++++++++++++++++++++++++

GLM-Spectrum classes and functions designed to work with GLM-Spectra computed from  MNE format sensorspace data

.. currentmodule:: osl.glm.glm_spectrum

.. autosummary::
   :toctree: stubs

   glm_spectrum

   group_glm_spectrum

   read_glm_spectrum

   plot_sensor_data
   
   plot_sensor_spectrum

   plot_joint_spectrum

   plot_joint_spectrum_clusters
   
   plot_channel_layout

   plot_with_cols

   decorate_spectrum

   prep_scaled_freq

   get_mne_sensor_cols

   SensorGLMSpectrum

   SensorGLMSpectrum.plot_joint_spectrum

   SensorGLMSpectrum.plot_sensor_spectrum

   GroupSensorGLMSpectrum

   GroupSensorGLMSpectrum.save_pkl

   GroupSensorGLMSpectrum.plot_joint_spectrum

   GroupSensorGLMSpectrum.get_fl_contrast

   MaxStatPermuteGLMSpectrum

   MaxStatPermuteGLMSpectrum.plot_sig_clusters

   ClusterPermuteGLMSpectrum

   ClusterPermuteGLMSpectrum.plot_sig_clusters
   

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