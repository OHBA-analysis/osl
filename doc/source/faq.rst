Frequently Asked Questions (FAQ)
================================

.. contents::
   :local:

If you  have a question that's not listed above, please submit an issue to the `GitHub repository <https://github.com/OHBA-analysis/osl/issues>`_. 

Installation
------------

How do I install osl?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended installation of the latest version via pip or github is described :doc:`here <install>`.



Preprocessing
-------------

How do I use the config API for preprocessing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The config is a text that is structured as follows:


.. code-block:: Python


   """
   meta: 
      event_codes:
         event_1: 1
         event_2: 2
   preproc:
      function_1: {argument_1: value, argument_2: value}
      function_2: {}
   """


``meta`` contains the events trigger codes, and ``preproc`` contains all the preprocessing functions/methods and the corresponding settings. The ``config`` can also be saved as yaml file - and loaded from it. It can also be represented as a Python dictionary.


How do I use a custom function in the pipeline?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a python functions, and make sure it is structured as follows:

.. code-block:: Python

   import osl

   def custom_function(dataset, userargs):
      # Get any arguments from the userargs dictionary
      target = userargs.pop("target", "raw")
      picks = userargs.pop("picks", "meg")
      # if a logger was set up, e.g., by run_proc_chain, log_or_print will write to the logfile
      osl.utils.log_or_print(f"Taking the absolute value of the {target} data.")

      # manipulate any keys in dataset here, for example using Raw.apply_function:
      dataset["rawabs"] = dataset[target].apply_function(np.abs, picks=picks)
   return dataset

   # add the function to the config
   config = """
   preproc:
      - custom_function: {picks: meg}
   """

   # supply the function to run_proc_chain / run_proc_batch
   osl.preprocessing.run_proc_chain(config, infile, subject, outdir, extra_funcs=[custom_function])

The custom function should have ``dataset`` and ``userargs`` input arguments, and a ``dataset`` output argument. Any key in dataset can be manipulated in place, or a new key can be added, in which case it will be saved according to the name of the key.
For example, if the data is saved as ``sub001-run01_preproc-raw.fif``, the ``rawabs`` key will be saved as ``sub001-run01_rawabs.fif``. 

The user can also print statements to an existing logfile using ``osl.utils.log_or_print``. 


How do I refine the pipeline for my data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Have a look at the :doc:`tutorials_build/preprocessing_automatic` tutorial.

What is MaxFilter and how do I use it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MaxFilter is Elekta licensed software, and is typically only used for Elekta/Megin data, though in principle it can be applied to other data source (incl. OPM's). It is used to remove external noise (e.g., environmental noise) and do head movement compensation. 
Maxfilter uses some extra reference sensors in the MEG together with Signal Space Seperation (SSS) to achieve this. MaxFilter has various settings, for which OSL has `wrappers <https://github.com/OHBA-analysis/osl/tree/main/osl/maxfilter>`_ for the 
Elekta software with some explanations of settings. Furthermore, `MNE-Python also has a maxfilter that doesn't require a license <https://mne.tools/stable/generated/mne.preprocessing.maxwell_filter.html>`_. Besides these references, also have a look at the 
`Maxfilter user manual <https://ohba-analysis.github.io/osl-docs/downloads/maxfilter_user_guide.pdf>`_ and at `these guidelines <https://lsr-wiki-01.mrc-cbu.cam.ac.uk/meg/maxpreproc>`_.



How can I preprocess my data using multiple cores (CPUs)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you're using ``osl.preprocessing.run_proc_batch`` you can parallelize the processing over datasets by using dask. This requires that you structure the main code of your Python file inside a ``if __name__ == '__main__':`` statement. 
We also need to start a Client and specify ``threads_per_worker=1`` and the number of cores to use (``n_workers``). Lastly, we need to specify ``dask_client=True`` in ``run_proc_batch``.

.. warning::

   ``threads_per_worker`` should always be set to 1. n_workers depends on your computing infrastructure. For example, if you’re on a personal computer with 8 cores, you can at most use ``n_workers=8``. If you’re working on a shared computing infrastructure, discuss the appropriate setting with your IT support. As a rule of thumb, here we will use half the cores that are available on your computer.

.. code-block:: Python

   # start a Dask Client
   from dask.distributed import Client
   client = Client(threads_per_worker=1, n_workers=4)


   if __name__ == '__main__':

      # write extra information here, e.g., definitions of config, files, output_dir

      osl.preprocessing.run_proc_batch(config, 
         inputs=infiles, 
         subjects=subjects_ids, 
         outdir=outdir, 
         dask_client=True)

How do I select which components to remove in ICA?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are several ways to identify artefact-related components. Comonly, components related to heartbeats and eyemovements (saccades/blinks) are removed.
These can be identified either automatically, e.g., by correlation with the ECG / EOG (when recorded), or manually, by inspecting the component topographies and timecourses. 
We recommend a combination of the two: have a automatic first pass, and manually adapting the labels where necessary. 

We provide command line functions in ``osl`` to do the manual checks and reject the components from the data post-hoc. See `ica_label <https://osl.readthedocs.io/en/latest/build/html/_modules/osl/preprocessing/ica_label.html#apply>`_

.. code-block:: Python

   (osl) > osl_ica_label None preprocessed sub001-ses01

Also see `Automnatic preprocessing using an OSL config <https://osl.readthedocs.io/en/latest/tutorials_build/preprocessing_automatic.html#manually-checking-ica>`_.

Regarding the manual detection, Eye and heart related components are usually quite easy to recognise. `this advise from the FieldTrip Toolbox is useful <https://www.fieldtriptoolbox.org/tutorial/ica_artifact_cleaning/#identifying-artifactual-components>`_: 
"Eye-related components are spatially localized on the frontal channels, blinks and vertical saccades are symmetric and horizontal saccades show a distinct left-right pattern. Heart-related components in MEG show up as a very deep source with a bipolar projecting over the left and right side of the helmet. It is common for both eye and heart components that you will see a few of them."
Note that you typically won't see Heart-related components in EEG. 


Source reconstruction
---------------------

How do I coregister my MRI and MEG data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This involves coregistering a number of different coordinate systems:

* MEG (Device) space - defined with respect to  the MEG dewar.
* Polhemus (Head) space - defined with respect to the locations of the fiducial locations (LPA, RPA and Nasion). The fiducial locations in polhemus space are typically acquired prior to the MEG scan, using a polhemus device.
* sMRI (Native) space - defined with respect to the structural MRI scan.
* MNI space - defined with respect to the MNI standard space brain.

See the :doc:`tutorials_build/preprocessing_automatic` tutorial to see how to coregister the data.

How do I use a custom function in the pipeline?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is done slightly differently than in the ``preprocessing`` module. Again, we need to define a python function, but the ``soure_recon`` module doesn't work with the ``dataset`` dictionary, so we might need to load/save data to disk directly.
As input arguments, we can use any input arguments that `run_src_chain <https://osl.readthedocs.io/en/latest/autoapi/osl/source_recon/index.html#osl.source_recon.batch.run_src_chain>`_ and `run_src_batch <https://osl.readthedocs.io/en/latest/autoapi/osl/source_recon/index.html#osl.source_recon.batch.run_src_batch>`_
take, such as ``subject``, ``outdir``, and ``smri_file``. We can also use ``userargs``, to specify any options you might want to supply in the config.
The user can also print statements to an existing logfile using ``osl.utils.log_or_print``. 

For example:

.. code-block:: Python

   import osl
   import numpy as np

   def fix_headshape_points(outdir, subject, userargs):
      filenames = osl.source_recon.rhino.get_coreg_filenames(outdir, subject)

      # Load saved headshape and nasion files
      hs = np.loadtxt(filenames["polhemus_headshape_file"])
      nas = np.loadtxt(filenames["polhemus_nasion_file"])
      lpa = np.loadtxt(filenames["polhemus_lpa_file"])
      rpa = np.loadtxt(filenames["polhemus_rpa_file"])

      # Remove headshape points on the nose
      remove = np.logical_and(hs[1] > max(lpa[1], rpa[1]), hs[2] < nas[2])
      hs = hs[:, ~remove]

      # Overwrite headshape file
      osl.utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
      np.savetxt(filenames["polhemus_headshape_file"], hs)


   # add the function to the config
   config = """
   source_recon:
      - fix_headshape_points: {}
   """

   # supply the function to run_src_chain / run_src_batch
   osl.source_recon.run_src_chain(config, infile, subject, outdir, smri_file, extra_funcs=[fix_headshape_points])


How do I use multiple cores for parallel preprocessing my data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`tutorials_build/preprocessing_automatic`

How do I refine the coregistration of a particular subject?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the `Deleting Headshape Points <https://osl.readthedocs.io/en/latest/tutorials_build/source-recon_deleting-headshape-points.html>`_ tutorial.



Utilities
---------

How do I use the Study class for finding my data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `Study <https://osl.readthedocs.io/en/latest/autoapi/osl/utils/index.html#osl.utils.study.Study>`_ class enables finding data paths with multiple wild cars, and selecting those that satisfy a specific wild card.

For example 

.. code-block:: Python

   import osl

   study = osl.utils.Study('/path/to/sub{subject_id}-run{run_id}_preproc-raw.fif')

   all_files = study.get()
   subject1_files = study.get(subject_id=1)


Other
-----


I found a bug, what do I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an issue `here <https://github.com/OHBA-analysis/osl/issues>`_.

Does osl contain functionality for training generative models (e.g., HMM, DyNeMo)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

osl does not contain functionality for training generative models, but we have developed another Python package, osl-dynamics, which contains functionality for training generative models. You can find osl-dynamics `here <https://github.com/OHBA-analysis/osl-dynamics>`_, and the documentation `here <https://osl-dynamics.readthedocs.io/en/latest/>`_.

Citing ``osl``
--------------
How can I cite the package?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For up-to-date citation information, please have a look at the citation information on `GitHub   <https://github.com/OHBA-analysis/osl/blob/main/CITATION.cff>`_ (Look for the button "Cite this repository"). 
Don't forget to also cite `MNE-Python <https://github.com/mne-tools/mne-python>`_, and, if you've used the `osl.source_recon` module, `FSL <https://fsl.fmrib.ox.ac.uk/fsl/docs/#/license>`_

