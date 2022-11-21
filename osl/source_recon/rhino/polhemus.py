"""Functions related to polhemus fiducials.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
import matplotlib.pyplot as plt

from mne.io import read_info
from mne.io.constants import FIFF


def extract_polhemus_from_info(
    fif_file,
    headshape_outfile,
    nasion_outfile,
    rpa_outfile,
    lpa_outfile,
    include_eeg_as_headshape=False,
    include_hpi_as_headshape=True,
):
    """Extract polhemus from FIF info.

    Extract polhemus fids and headshape points from MNE raw.info and write them out
    in the required file format for rhino (in head/polhemus space in mm). Should only
    be used with MNE-derived .fif files that have the expected digitised points held
    in info['dig'] of fif_file.

    Parameters
    ----------
    fif_file : string
        Full path to MNE-derived fif file.
    headshape_outfile : string
        Filename to save naison to.
    nasion_outfile : string
        Filename to save naison to.
    rpa_outfile : string
        Filename to save naison to.
    lpa_outfile : string
        Filename to save naison to.
    include_eeg_as_headshape : bool
        Should we include EEG locations as headshape points?
    include_hpi_as_headshape : bool
        Should we include HPI locations as headshape points?
    """
    # Lists to hold polhemus data
    polhemus_headshape = []
    polhemus_rpa = []
    polhemus_lpa = []
    polhemus_nasion = []

    # Read info from fif file
    info = read_info(fif_file)
    for dig in info["dig"]:

        # Check dig is in HEAD/Polhemus space
        if dig["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
            raise ValueError("{} is not in Head/Polhemus space".format(dig["ident"]))

        if dig["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            if dig["ident"] == FIFF.FIFFV_POINT_LPA:
                polhemus_lpa = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_RPA:
                polhemus_rpa = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_NASION:
                polhemus_nasion = dig["r"]
            else:
                raise ValueError("Unknown fiducial: {}".format(dig["ident"]))
        elif dig["kind"] == FIFF.FIFFV_POINT_EXTRA:
            polhemus_headshape.append(dig["r"])
        elif dig["kind"] == FIFF.FIFFV_POINT_EEG and include_eeg_as_headshape:
            polhemus_headshape.append(dig["r"])
        elif dig["kind"] == FIFF.FIFFV_POINT_HPI and include_hpi_as_headshape:
            polhemus_headshape.append(dig["r"])

    # Save
    np.savetxt(nasion_outfile, polhemus_nasion * 1000)
    np.savetxt(rpa_outfile, polhemus_rpa * 1000)
    np.savetxt(lpa_outfile, polhemus_lpa * 1000)
    np.savetxt(headshape_outfile, np.array(polhemus_headshape).T * 1000)


def plot_polhemus_points(
    txt_fnames, colors=None, scales=None, markers=None, alphas=None
):
    plt.figure()
    ax = plt.axes(projection="3d")
    for ss in range(len(txt_fnames)):
        if alphas is None:
            alpha = 1
        else:
            alpha = alphas[ss]
        if colors is None:
            color = (0.5, 0.5, 0.5)
        else:
            color = colors[ss]
        if scales is None:
            scale = 10
        else:
            scale = scales[ss]
        if markers is None:
            marker = 1
        else:
            marker = markers[ss]

        pnts = np.loadtxt(txt_fnames[ss])
        ax.scatter(
            pnts[0],
            pnts[1],
            pnts[2],
            color=color,
            s=scale,
            alpha=alpha,
            marker=marker,
        )
