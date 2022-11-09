#!/usr/bin/env python

"""Reporting tool for source reconstruction.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import os.path as op
import pickle
from pathlib import Path
from shutil import copy

import numpy as np

from . import raw_report
from ..source_recon import rhino, parcellation, sign_flipping


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
        subject_data = {}

    # Data for the report
    data = {}
    data["fif_id"] = subject
    data["filename"] = subject

    # What have we done for this subject?
    data["coregister"] = subject_data.pop("coregister", False)
    data["beamform"] = subject_data.pop("beamform", False)
    data["beamform_and_parcellate"] = subject_data.pop("beamform_and_parcellate", False)
    data["fix_sign_ambiguity"] = subject_data.pop("fix_sign_ambiguity", False)

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
    panel_template = raw_report.load_template('src_panel')

    for i in range(total):
        panels.append(panel_template.render(data=data[i]))

    # Hyperlink to each panel on the page
    filenames = ""
    for i in range(total):
        filename = Path(data[i]["filename"]).name
        filenames += "{0}. {1}<br>".format(i + 1, filename)

    # Render the full page
    page_template = raw_report.load_template('report')
    page = page_template.render(panels=panels, filenames=filenames)

    # Write the output file
    outpath = Path(reportdir) / 'report.html'
    with open(outpath, 'w') as f:
        f.write(page)

    return True


def plot_parcellation(parcellation_file, reportdir, subject):
    """Plots parcellation."""
    output_file = reportdir / subject / "parc.png"
    parcellation.plot_parcellation(parcellation_file, output_file=output_file)
    return f"{subject}/parc.png"


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
