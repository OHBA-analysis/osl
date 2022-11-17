"""Examle script for manual ICA artefact rejection.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import preprocessing


preproc_dir = "/ohba/pi/knobre/cgohil/dg_int_ext/preproc"
subjects = ["s01_block_01", "s01_block_02"]

run_ids = []
preproc_files = []
for subject in subjects:
    run_id = f"InEx_{subject}_tsss"
    run_ids.append(run_id)
    preproc_files.append(
        f"{preproc_dir}/{run_id}/{run_id}_preproc_raw.fif"
    )

for preproc_file, run_id in zip(preproc_files, run_ids):
    # Load raw fif, events and ICA
    dataset = preprocessing.read_dataset(preproc_file)
    raw = dataset["raw"]
    ica = dataset["ica"]

    # Mark bad ICA components interactively
    preprocessing.plot_ica(ica, raw)

    # Apply ICA
    raw = ica.apply(raw)

    # Save cleaned data
    dataset["raw"] = raw
    dataset["ica"] = ica
    preprocessing.wrte_dataset(
        dataset, preproc_dir, run_id, overwrite=True
    )
