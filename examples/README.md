# Example scripts

One directory per scanner, each a pipeline from the raw recordings to
sign-flipped parcel time courses: `elekta/`, `ctf/`, `eeg/` and `opm/`,
and `oxford/` with the pipelines for the OHBA scanner (automatic ICA,
manual ICA, SSP, no plots, MaxFilter alone).
Every script has the same shape: the sessions (a list of dicts, or a CSV
file with the same columns) and a config at the top, then `run_batch`,
which runs the config on every session in parallel, writes a log per
session and the QC report (`outdir/qc/report.html`). The stages are
separate scripts so the report can be checked between them, but their
sections can be put in one config and run in one go.

`misc/` has smaller scripts for particular situations. `v2/` holds the
scripts written for osl-ephys 2.4, which are not updated (the toolbox
paper, the Oxford pipelines as they were, and an MNE-only walkthrough);
the release notes of v3.0.0 list what changed.

For example scripts on public M/EEG datasets, see
https://github.com/OHBA-analysis/OHBA-Examples.
