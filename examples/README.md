# Example scripts

One directory per scanner, each a pipeline from the raw recordings to
sign-flipped parcel time courses: `elekta/`, `ctf/`, `eeg/` and `opm/`.
Every script has the same shape: the sessions (a list of dicts, or a CSV
file with the same columns) and a config at the top, then `run_batch`,
which runs the config on every session in parallel, writes a log per
session and the QC report (`outdir/qc/report.html`). The stages are
separate scripts so the report can be checked between them, but their
sections can be put in one config and run in one go.

`misc/` has smaller scripts for particular situations. `v2/` holds the
scripts written for osl-ephys 2.4, which are not updated (the toolbox paper
and the Oxford pipelines); see the migration page of the documentation for
the differences.

For example scripts on public M/EEG datasets, see
https://github.com/OHBA-analysis/OHBA-Examples.
