.. raw:: html

   <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1em;">
     <h1 style="margin: 0;">OSL Electrophysiology Toolbox</h1>
     <div>
       <a href="https://www.oxcin.ox.ac.uk/about/our-locations/OHBA" target="_blank">
         <img src="https://raw.githubusercontent.com/OHBA-analysis/osl-files/main/docs/logo/ohba-logo.png" width="125px">
       </a>
       <a href="https://www.oxcin.ox.ac.uk" target="_blank">
         <img src="https://raw.githubusercontent.com/OHBA-analysis/osl-files/main/docs/logo/oxcin-logo.png" width="220px">
       </a>
     </div>
   </div>

osl-ephys is a Python toolbox for analysing electrophysiological data (MEG, EEG and OPM) from the raw recording to parcellated source-space time courses. It builds on `MNE-Python <https://mne.tools>`_ and adds:

- Preprocessing driven by a YAML config, run on one session or batched over a dataset in parallel.
- A MaxFilter wrapper for Elekta/MEGIN data.
- Coregistration and volumetric source reconstruction with RHINO: surface extraction with FSL, coregistration to the head shape, forward modelling, LCMV beamforming and minimum norm.
- Parcellation with spatial leakage correction and sign flipping across sessions.
- Quality control reports across sessions.
- Sensor-level statistics with GLM-Spectrum and GLM-Epochs permutation testing.

For more information on how to use osl-ephys see the :doc:`documentation <documentation>`.

If you would like to request new features or if you're confident that you have found a bug, please create a new issue on the `GitHub issues <https://github.com/OHBA-analysis/osl-ephys/issues>`_ page.

This package was developed by the Oxford Centre for Human Brain Activity (OHBA) Methods Group at the University of Oxford. Our group website is `here <https://www.psych.ox.ac.uk/research/ohba-analysis-group>`_.

If you find this toolbox useful, please cite the `paper <https://doi.org/10.3389/fnins.2025.1522675>`_:

    **van Es, M. W. J., Gohil, C., Quinn, A. J., & Woolrich, M. W. (2025). osl-ephys: a Python toolbox for the analysis of electrophysiology data. Frontiers in Neuroscience, 19, 1522675.**

osl-ephys builds on MNE-Python, and the source reconstruction on FSL. Please cite those packages as well when you use them.

-----------------------

Contents
========

.. toctree::
   :maxdepth: 2

   Install <install>
   Documentation <documentation>
   Migrating from v2 <migration>
   FAQ <faq>
   API Reference <autoapi/osl_ephys/index>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
