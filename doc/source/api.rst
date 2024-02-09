API
===

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


GLM
***

GLM Base
++++++++

Modality specific wrappers for glmtools.

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
++++++++++

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
++++++++++++

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


RHINO
*****

Tools for Coregistration and forward modeling 

Coregistration
++++++++++++++

Deep-level RHINO functions used by wrappers

.. currentmodule:: osl.source_recon.rhino.coreg

.. autosummary::
   :toctree: stubs

    bem_display
    
    coreg

    coreg_display
    
    coreg_metrics
    
    get_coreg_filenames


Forward modeling
++++++++++++++++

Deep-level RHINO functions used by wrappers

.. currentmodule:: osl.source_recon.rhino.forward_model

.. autosummary::
   :toctree: stubs

    forward_model

    make_fwd_solution

    setup_volume_source_space


FSL-utils
+++++++++

Deep-level RHINO functions used by wrappers

.. currentmodule:: osl.source_recon.rhino.fsl_utils

.. autosummary::
   :toctree: stubs

    check_fsl

    fsleyes

    fsleyes_overlay
    
    setup_fsl


Polhemus
++++++++

Deep-level RHINO functions used by wrappers

.. currentmodule:: osl.source_recon.rhino.polhemus

.. autosummary::
   :toctree: stubs

    delete_headshape_points
    
    extract_polhemus_from_info

    plot_polhemus_points


Surfaces
++++++++

Deep-level RHINO functions used by wrappers

.. currentmodule:: osl.source_recon.rhino.surfaces

.. autosummary::
   :toctree: stubs

    check_if_already_computed

    compute_surfaces
    
    get_surfaces_filenames

    plot_surfaces

    surfaces_display


Utils
+++++

Deep-level RHINO functions used by wrappers

.. currentmodule:: osl.source_recon.rhino.utils

.. autosummary::
   :toctree: stubs

    best_fit_transform

    extract_rhino_files
    
    get_gridstep
    
    get_rhino_files

    icp

    majority

    nearest_neighbor
    
    niimask2indexpointcloud

    niimask2mmpointcloud

    recon_timeseries2niftii

    rhino_icp
    
    rigid_transform_3D

    save_or_show_renderer

    system_call

    xform_points

    _closest_node

    _create_freesurfer_mesh_from_bet_surface

    _create_freesurfer_meshes_from_bet_surfaces

    _get_flirtcoords2native_xform

    _get_vol_info_from_nii

    _get_sform

    _get_mne_xform_from_flirt_xform

    _get_flirt_xform_between_axes

    _get_mni_sform

    _get_orient

    _get_vtk_mesh_native

    _binary_majority3d

    _timeseries2nii

    _transform_bet_surfaces

    _transform_vtk_mesh


Source Reconstruction
*********************

Pipeline Functions
++++++++++++++++++

Primary user-level functions for running OSL coregistration and source_recon functions.

.. currentmodule:: osl.source_recon.batch

.. autosummary::
   :toctree: stubs

    run_src_chain

    run_src_batch


Wrappers
++++++++

Primary wrapper functions to use in a source_recon configuration

.. currentmodule:: osl.source_recon.wrappers

.. autosummary::
   :toctree: stubs

    beamform

    beamform_and_parcellate
    
    compute_surfaces

    compute_surfaces_coregister_and_forward_model

    coregister

    extract_fiducials_from_fif

    extract_rhino_files

    find_template_subject

    fix_sign_ambiguity

    forward_model

    parcellate


Beamforming
+++++++++++

Second-level beamforming functions used by wrappers

.. currentmodule:: osl.source_recon.beamforming

.. autosummary::
   :toctree: stubs

    apply_lcmv

    apply_lcmv_raw

    get_beamforming_filenames

    get_recon_timeseries

    load_lcmv

    make_lcmv

    make_plots

    transform_leadfield

    transform_recon_timeseries

    voxel_timeseries

    _compute_beamformer
    
    _make_lcmv

    _prepare_beamformer_input


Sign-flipping
+++++++++++++

Second-level sign-flipping functions used by wrappers

.. currentmodule:: osl.source_recon.sign_flipping

.. autosummary::
   :toctree: stubs
    
    apply_flips
    
    apply_flips_to_covariance

    covariance_matrix_correlation

    find_flips

    find_template_subject

    load_covariances

    randomly_flip

    std_data

    time_embed

    _get_parc_chans


Utils
+++++

Utility functions for running an OSL Source Recon pipeline.

.. currentmodule:: osl.source_recon.batch

.. autosummary::
   :toctree: stubs

    load_config

    find_func


Parcellation
************

Second-level Parcellation functions used by wrappers

.. currentmodule:: osl.source_recon.parcellation.parcellation

.. autosummary::
   :toctree: stubs

   convert2mne_raw

   convert2mne_epochs
   
   convert2niftii

   find_file
   
   load_parcellation

   parcel_centers

   parcellate_timeseries

   plot_correlation
   
   plot_parcellation

   plot_psd

   spatial_dist_adjacency

   symmetric_orthogonalise

   _get_parcel_timeseries

   _parcel_timeseries2nii

   _resample_parcellation


Nifti Utils
++++++++++++

Second-level Parcellation functions used by wrappers

.. currentmodule:: osl.source_recon.parcellation.nii

.. autosummary::
   :toctree: stubs

    append_4d_parcellation
    
    convert_3dparc_to_4d
    
    convert_4dparc_to_3d

    spatially_downsample


Report
******

Sensor level
++++++++++++

.. currentmodule:: osl.report.preproc_report

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
++++++

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
