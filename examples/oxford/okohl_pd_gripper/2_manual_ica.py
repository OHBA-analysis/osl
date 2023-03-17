"""Apply manual ICA to preprocessed sensor data.

"""

# Authors: Oliver Kohl <oliver.kohl@psych.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import mne
import matplotlib.pyplot as plt

from osl import preprocessing


def check_ica(raw, ica, save_dir):
    """Checks ICA."""

    # Make output directory
    os.makedirs(save_dir, exist_ok=True)

    # Find EOG and ECG correlations
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
    ica_indices = ecg_indices + eog_indices
    ica_scores = ecg_scores + eog_scores

    # Barplot of ICA component "EOG match" and "ECG match" scores
    correlation_plot = ica.plot_scores(ica_scores)
    plt.savefig(save_dir + "/correl_plot.png", bbox_inches="tight")
    plt.close()

    # Plot bad components
    bad_component_plot = ica.plot_components(ica.exclude)
    plt.savefig(save_dir + "/bad_components.png", bbox_inches="tight")
    plt.close()


# Preprocessing directory and subjects
preproc_dir = "/ohba/pi/knobre/cgohil/pd_gripper/preproc"
subjects = ["HC01", "HC02"]

# Check ICA
for subject in subjects:
    run_id = f"{subject}_gripper_trans"
    report_dir = f"{preproc_dir}/report_post_manual_ICA/{run_id}"

    # Load preprocessed fif and ICA
    dataset = preprocessing.read_dataset(preproc_file, preload=True)
    raw = dataset["raw"]
    ica = dataset["ica"]

    # Mark bad ICA components interactively
    preprocessing.plot_ica(ica, raw)

    #get_ipython().run_line_magic('matplotlib', 'qt')
    #ica.plot_sources(raw,title='ICA');
    #raw.plot();

    # Create figures
    check_ica(raw, ica, report_dir)
    plt.close("all")

    # Apply ICA
    raw = ica.apply(raw, exclude=ica.exclude)

    # Plot power spectra of cleaned data
    raw.plot_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4))
    plt.savefig(report_dir + "/powspec.png", bbox_inches="tight")
    plt.close()

    # Save cleaned data
    dataset["raw"].save(preproc_file, overwrite=True)
