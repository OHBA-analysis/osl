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

    # Data to include in the report
    data = {}

    # Add info in config to data
    for stage in config["source_recon"]:
        data.update(stage)

    # Subject info
    data["fif_id"] = subject
    data["filename"] = subject

    # Coregistration plots
    data["plt_coreg"] = plot_coreg(reportdir, src_dir, subject)

    # Beamforming plots
    for name in ["cov", "svd"]:
        rhino_plot = op.join(src_dir, subject, "rhino", f"filter_{name}.png")
        if Path(rhino_plot).exists():
            report_plot = op.join(reportdir, subject, f"filter_{name}.png")
            copy(rhino_plot, report_plot)
            data[f"plt_filter_{name}"] = op.join(subject, f"filter_{name}.png")

    if "beamform_and_parcellate" in data:
        # Parcellation info
        data["filepath"] = src_dir / subject / "rhino/parc.npy"

        parcel_ts = np.load(data["filepath"])
        data["n_samples"] = parcel_ts.shape[0]
        data["n_parcels"] = parcel_ts.shape[1]

        # Parcellation plots
        data["plt_parc"] = plot_parcellation(
            data["beamform_and_parcellate"]["parcellation_file"], reportdir, subject
        )

    if "fix_sign_ambiguity" in data:
        # sflip info
        sflip_info_filepath = src_dir / subject / "rhino/sflip_info.pkl"
        sflip_info = pickle.load(open(sflip_info_filepath, "rb"))

        data["sflip"] = {"metrics": np.around(sflip_info["metrics"], decimals=3)}

        # sflip plots
        data["plt_sflip"] = plot_sign_flipping(
            sflip_info, reportdir, subject
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


def plot_coreg(reportdir, src_dir, subject):
    """Plot coregistration."""
    rhino.coreg_display(
        src_dir,
        subject,
        filename=reportdir / subject / "coreg.html",
        display_outskin_with_nose=False,
    )
    return f"{subject}/coreg.html"


def plot_parcellation(parcellation_file, reportdir, subject):
    """Plots parcellation."""
    output_file = reportdir / subject / "parc.png"
    parcellation.plot_parcellation(parcellation_file, output_file=output_file)
    return f"{subject}/parc.png"

def plot_sign_flipping(sflip_info, reportdir, subject):
    """Plots sign flipping."""
    output_file = reportdir / subject / "sflip.png"

    sign_flipping.plot_sign_flipping(
        sflip_info['cov'],
        sflip_info['template_cov'],
        sflip_info['n_embeddings'],
        sflip_info['flips'],
        output_file)

    return f"{subject}/sflip.png"