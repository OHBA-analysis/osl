# Pipeline without plots

The whole pipeline in one config and one run, without the QC report and
its plots, for a computer where plotting fails (no display, no working 3D
rendering). `qc_dir=False` turns the report off; the terminal equivalent
is `osl-ephys run config.yml sessions.csv data --no-report`. The ICA
components are labelled automatically, since there is nothing to review.

The 3D coregistration plot is the usual culprit; on a headless Linux
server it can be kept instead by installing the software-rendering build
of VTK (see the installation page of the documentation).
