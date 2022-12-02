"""Reporting tool for source reconstruction.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import os.path as op
from pathlib import Path
from shutil import copy

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from . import raw_report
from ..source_recon import parcellation


def gen_html_data(config, src_dir, subject, reportdir, logger=None):
    """Generate data for HTML report.

    Parameters
    ----------
    config : dict
        Source reconstruction config.
    src_dir : str
        Source reconstruction directory.
    subject : str
        Subject name.
    reportdir : str
        Report directory.
    logger : logging.getLogger
        Logger.
    """
    src_dir = Path(src_dir)
    reportdir = Path(reportdir)

    # Make directory for plots contained in the report
    os.makedirs(reportdir, exist_ok=True)
    os.makedirs(reportdir / subject, exist_ok=True)

    # Open data saved by the source_recon.wrappers
    data_file = f"{src_dir}/{subject}/report_data.pkl"
    if Path(data_file).exists():
        subject_data = pickle.load(open(data_file, "rb"))
    else:
        return

    # Check if this function has been called before
    if Path(f"{reportdir}/{subject}/data.pkl").exists():
        # Load the data object from last time
        data = pickle.load(open(f"{reportdir}/{subject}/data.pkl", "rb"))

        if "config" in data:
            # Update the config based on this run
            data["config"] = update_config(data["config"], config)

    # Otherwise, this is the first time this has been called
    else:
        # Data for the report
        data = {}
        data["config"] = config

    data["fif_id"] = subject
    data["filename"] = subject

    # What have we done for this subject?
    data["coreg"] = subject_data.pop("coreg", False)
    data["beamform"] = subject_data.pop("beamform", False)
    data["beamform_and_parcellate"] = subject_data.pop("beamform_and_parcellate", False)
    data["fix_sign_ambiguity"] = subject_data.pop("fix_sign_ambiguity", False)

    # Save info
    if data["beamform_and_parcellate"]:
        data["n_samples"] = subject_data["n_samples"]
    if data["coreg"]:
        data["fid_err"] = subject_data["fid_err"]
    if data["beamform_and_parcellate"]:
        data["parcellation_file"] = subject_data["parcellation_file"]
        data["parcellation_filename"] = Path(subject_data["parcellation_file"]).name
    if data["fix_sign_ambiguity"]:
        data["template"] = subject_data["template"]
        data["metrics"] = subject_data["metrics"]

    # Copy plots
    if "coreg_plot" in subject_data:
        data["plt_coreg"] = f"{subject}/coreg.html"
        copy(subject_data["coreg_plot"], f"{reportdir}/{subject}/coreg.html")

    if "filter_cov_plot" in subject_data:
        data["plt_filter_cov"] = f"{subject}/filter_cov.png"
        copy(subject_data["filter_cov_plot"], f"{reportdir}/{subject}/filter_cov.png")

    if "filter_svd_plot" in subject_data:
        data["plt_filter_svd"] = f"{subject}/filter_svd.png"
        copy(subject_data["filter_svd_plot"], f"{reportdir}/{subject}/filter_svd.png")

    if "filter_svd_plot" in subject_data:
        data["plt_filter_svd"] = f"{subject}/filter_svd.png"
        copy(subject_data["filter_svd_plot"], f"{reportdir}/{subject}/filter_svd.png")

    if "parc_corr_plot" in subject_data:
        data["plt_parc_corr"] = f"{subject}/parc_corr.png"
        copy(subject_data["parc_corr_plot"], f"{reportdir}/{subject}/parc_corr.png")

    # Save data in the report directory
    pickle.dump(data, open(f"{reportdir}/{subject}/data.pkl", "wb"))


def gen_html_page(reportdir):
    """Generate an HTML page from a report directory.

    Parameters
    ----------
    reportdir : str
        Directory to generate HTML report with.
    """
    reportdir = Path(reportdir)

    # Subdirectories which contains plots for each fif file
    subdirs = sorted(
        [d.stem for d in Path(reportdir).iterdir() if d.is_dir()]
    )

    # Load HTML data
    data = []
    for subdir in subdirs:
        subdir = Path(subdir)
        # Just generate the html page with the successful runs
        try:
            data.append(pickle.load(open(reportdir / subdir / "data.pkl", "rb")))
        except:
            pass

    total = len(data)
    if total == 0:
        return False

    # Add info to data indicating the total number of files
    # and an id for each file
    for i in range(total):
        data[i]["num"] = i + 1
        data[i]["total"] = total

    # Create panels
    panels = []
    panel_template = raw_report.load_template('src_subject_panel')

    for i in range(total):
        panels.append(panel_template.render(data=data[i]))

    # Hyperlink to each panel on the page
    filenames = ""
    for i in range(total):
        filename = Path(data[i]["filename"]).name
        filenames += "{0}. {1}<br>".format(i + 1, filename)

    # Render the full page
    page_template = raw_report.load_template('subject_report')
    page = page_template.render(panels=panels, filenames=filenames)

    # Write the output file
    outpath = Path(reportdir) / 'subject_report.html'
    with open(outpath, 'w') as f:
        f.write(page)

    return True


def gen_html_summary(reportdir):
    """Generate an HTML summary from a report directory.

    Parameters
    ----------
    reportdir : str
        Directory to generate HTML summary report with.
    """
    reportdir = Path(reportdir)

    # Subdirectories which contains plots for each fif file
    subdirs = sorted(
        [d.stem for d in Path(reportdir).iterdir() if d.is_dir()]
    )

    # Load HTML data
    subject_data = []
    for subdir in subdirs:
        subdir = Path(subdir)
        # Just generate the html page with the successful runs
        try:
            subject_data.append(
                pickle.load(open(reportdir / subdir / "data.pkl", "rb"))
            )
        except:
            pass

    total = len(subject_data)
    if total == 0:
        return False

    # Data used in the summary report
    data = {}
    data["total"] = total
    data["config"] = subject_data[0]["config"]
    data["coreg"] = subject_data[0]["coreg"]
    data["beamform"] = subject_data[0]["beamform"]
    data["beamform_and_parcellate"] = subject_data[0]["beamform_and_parcellate"]
    data["fix_sign_ambiguity"] = subject_data[0]["fix_sign_ambiguity"]

    if data["coreg"]:
        subjects = np.array([d["filename"] for d in subject_data])

        fid_err_table = {
            "subjects": [], "nas_err": [], "lpa_err": [], "rpa_err": [],
        }
        for d in subject_data:
            if "fid_err" in d:
                if d["fid_err"] is not None:
                    fid_err_table["subjects"].append(d["fif_id"])
                    fid_err_table["nas_err"].append(np.round(d["fid_err"][0], decimals=2))
                    fid_err_table["lpa_err"].append(np.round(d["fid_err"][1], decimals=2))
                    fid_err_table["rpa_err"].append(np.round(d["fid_err"][2], decimals=2))
        if len(fid_err_table["subjects"]) > 1:
            data["coreg_table"] = tabulate(
                np.c_[
                    fid_err_table["subjects"],
                    fid_err_table["nas_err"],
                    fid_err_table["lpa_err"],
                    fid_err_table["rpa_err"],
                ],
                tablefmt="html",
                headers=["Subject", "Nasion", "LPA", "RPA"],
            )

    # Create plots
    os.makedirs(f"{reportdir}/summary", exist_ok=True)

    data["plt_config"] = plot_config(data["config"], reportdir)

    if "parcellation_file" in subject_data[0]:
        data["parcellation_filename"] = subject_data[0]["parcellation_filename"]
        data["plt_parc"] = plot_parcellation(
            subject_data[0]["parcellation_file"],
            reportdir,
        )

    if data["fix_sign_ambiguity"]:
        data["template"] = subject_data[0]["template"]
        metrics = np.array([d["metrics"] for d in subject_data if "metrics" in d])
        data["plt_sflip"] = plot_sign_flipping_results(metrics, reportdir)

    # Create panel
    panel_template = raw_report.load_template('src_summary_panel')
    panel = panel_template.render(data=data)

    # List of filenames
    filenames = ""
    for i in range(total):
        filename = Path(subject_data[i]["filename"]).name
        filenames += "{0}. {1}<br>".format(i + 1, filename)

    # Render the full page
    page_template = raw_report.load_template('summary_report')
    page = page_template.render(panel=panel, filenames=filenames)

    # Write the output file
    outpath = Path(reportdir) / 'summary_report.html'
    with open(outpath, 'w') as f:
        f.write(page)

    return True


def plot_config(config, reportdir):
    """Plots a config flowchart.

    Parameters
    ----------
    config : dict
        Config to plot.
    reportdir : str
        Path to report directory. We will save the plot in this directory.

    Returns
    -------
    path : str
        Path to plot.
    """

    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(top=0.95, bottom=0.05)
    ax = plt.subplot(111, frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])

    stage_height = 1 / (1 + len(config["source_recon"]))
    stagecol = "wheat"
    startcol = "red"

    box = dict(boxstyle="round", facecolor=stagecol, alpha=1, pad=0.3)
    startbox = dict(boxstyle="round", facecolor=startcol, alpha=1)
    font = {
        "family": "serif",
        "color": "k",
        "weight": "normal",
        "size": 14,
    }

    stages = [{"input": ""}, *config["source_recon"], {"output": ""}]
    stage_str = "$\\bf{{{0}}}$ {1}"

    ax.arrow(
        0.5, 1, 0.0, -1, fc="k", ec="k", head_width=0.045,
        head_length=0.035, length_includes_head=True,
    )

    for idx, stage in enumerate(stages):
        method, userargs = next(iter(stage.items()))

        method = method.replace("_", "\_")
        if method in ["input", "output"]:
            b = startbox
        else:
            b = box
            method = method + ":"

        ax.text(
            0.5,
            1 - stage_height * idx,
            stage_str.format(method, str(userargs)[1:-1]),
            ha="center",
            va="center",
            bbox=b,
            fontdict=font,
            wrap=True,
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.25, 0.75)

    fig.savefig(f"{reportdir}/summary/config.png", dpi=300, transparent=True)
    plt.close(fig)

    return f"summary/config.png"


def plot_parcellation(parcellation_file, reportdir):
    """Plot parcellation.

    Parameters
    ----------
    parcellation_file : str
        Path to parcellation file.
    reportdir : str
        Path to report directory. We will save the plot in this directory.

    Returns
    -------
    path : str
        Path to plot.
    """
    output_file = reportdir / "summary/parc.png"
    parcellation.plot_parcellation(parcellation_file, output_file=output_file)
    return f"summary/parc.png"


def plot_sign_flipping_results(metrics, reportdir):
    """Plot sign flipping results.

    Parameters
    ----------
    metrics : np.ndarray
        Sign flipping metrics. Shape is (n_subjects, n_iter + 1).
    reportdir : str
        Path to report directory. We will save the plot in this directory.

    Returns
    -------
    path : str
        Path to plot.
    """
    output_file = reportdir / "summary/sflip.png"
    fig, ax = plt.subplots()
    for i in range(metrics.shape[-1]):
        ax.plot(
            range(1, metrics.shape[0] + 1),
            metrics[:, i],
            label=f"Init {i}",
        )
    ax.legend()
    ax.set_xlabel("Subject")
    ax.set_ylabel("Correlation w.r.t. template subject covariance")
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return f"summary/sflip.png"


def add_to_data(data_file, info):
    """Adds info to a dictionary containing info for the source recon report.

    Parameters
    ----------
    data_file : str
        Path to pickle file containing the data dictionary.
    info : dict
        Info to add.
    """
    if Path(data_file).exists():
        data = pickle.load(open(data_file, "rb"))
    else:
        data = {}
    data.update(info)
    pickle.dump(data, open(data_file, "wb"))


def update_config(old_config, new_config):
    """Merge/update a config.

    Parameters
    ----------
    old_config : dict
        Old config.
    new_config : dict
        New config.

    Returns
    -------
    config : dict
        Merge/updated config.
    """
    old_stages = []
    for stage in old_config["source_recon"]:
        for k, v in stage.items():
            old_stages.append(k)
    new_stages = []
    for stage in new_config["source_recon"]:
        for k, v in stage.items():
            new_stages.append(k)
    for i, new_stage in enumerate(new_stages):
        if new_stage not in old_stages:
            old_config["source_recon"].append(new_config["source_recon"][i])
        else:
            for j in range(len(old_config["source_recon"])):
                if new_stage in old_config["source_recon"][j]:
                    old_config["source_recon"][j][new_stage] = (
                        new_config["source_recon"][i][new_stage]
                    )
    return old_config
