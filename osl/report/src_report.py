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
from ..source_recon import rhino, parcellation


def gen_html_data(config, src_dir, rhino_dir, subject, reportdir, logger=None):
    """Generate data for HTML report."""

    # Make directory for plots contained in the report
    os.makedirs(reportdir / subject, exist_ok=True)

    # Data to include in the report
    data = {}

    # Add info in config to data
    for stage in config["source_recon"]:
        data.update(stage)

    # Subject info
    data["fif_id"] = subject
    data["subject"] = subject
    data["filename"] = subject + ".npy"

    # Coregistration plots
    data["plt_coreg"] = plot_coreg(reportdir, rhino_dir, subject)

    # Beamforming plots
    for name in ["cov", "svd"]:
        rhino_plot = op.join(rhino_dir, subject, f"filter_{name}.png")
        if Path(rhino_plot).exists():
            report_plot = op.join(reportdir, subject, f"filter_{name}.png")
            copy(rhino_plot, report_plot)
            data[f"plt_filter_{name}"] = op.join(subject, f"filter_{name}.png")

    # Parcellation plots
    data["plt_parc"] = plot_parcellation(
        data["beamform_and_parcellate"]["parcellation_file"], reportdir, subject
    )

    # Save data that will be used to create html page
    with open(reportdir / subject / 'data.pkl', 'wb') as outfile:
        pickle.dump(data, outfile)


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


def plot_coreg(reportdir, rhino_dir, subject):
    """Plot coregistration."""
    rhino.coreg_display(rhino_dir, subject, filename=reportdir / subject / "coreg.html")
    return f"{subject}/coreg.html"


def plot_parcellation(parcellation_file, reportdir, subject):
    """Plots parcellation."""
    output_file = reportdir / subject / "parc.png"
    parcellation.plot_parcellation(parcellation_file, output_file=output_file)
    return f"{subject}/parc.png"
