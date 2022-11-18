"""Example script for manual ICA artefact rejection.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl import preprocessing

# Setup paths to preprocessed data files
preproc_dir = "/ohba/pi/knobre/cgohil/dg_int_ext/preproc"
subjects = ["s01_block_01", "s01_block_02"]

run_ids = []
preproc_files = []
for subject in subjects:
    run_id = f"InEx_{subject}_tsss"
    run_ids.append(run_id)
    preproc_files.append(f"{preproc_dir}/{run_id}/{run_id}_preproc_raw.fif")

# Manual ICA artefact rejection
for preproc_file, run_id in zip(preproc_files, run_ids):

    # Load preprocessed fif and ICA
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
    preprocessing.write_dataset(dataset, preproc_dir, run_id, overwrite=True)
