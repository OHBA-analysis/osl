"""Functions related to polhemus fiducials.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>

import numpy as np
import matplotlib.pyplot as plt

from mne.io import read_info
from mne.io.constants import FIFF

import sys
from osl.source_recon.rhino.coreg import get_coreg_filenames
from osl.utils.logger import log_or_print

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

    Extract polhemus fids and headshape points from MNE raw.info and write them out in the required file format for rhino (in head/polhemus space in mm).
    Should only be used with MNE-derived .fif files that have the expected digitised points held in info['dig'] of fif_file.

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
    log_or_print("Extracting polhemus from fif info")

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
    log_or_print(f"saved: {nasion_outfile}")
    np.savetxt(nasion_outfile, polhemus_nasion * 1000)
    log_or_print(f"saved: {rpa_outfile}")
    np.savetxt(rpa_outfile, polhemus_rpa * 1000)
    log_or_print(f"saved: {lpa_outfile}")
    np.savetxt(lpa_outfile, polhemus_lpa * 1000)
    log_or_print(f"saved: {headshape_outfile}")
    np.savetxt(headshape_outfile, np.array(polhemus_headshape).T * 1000)


def plot_polhemus_points(txt_fnames, colors=None, scales=None, markers=None, alphas=None):
    """Plot polhemus points.
    
    Parameters
    ----------
    txt_fnames : list of strings
        List of filenames containing polhemus points.
    colors : list of tuples
        List of colors for each set of points.
    scales : list of floats
        List of scales for each set of points.
    markers : list of strings
        List of markers for each set of points.
    alphas : list of floats
        List of alphas for each set of points.
    """
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
        ax.scatter(pnts[0], pnts[1], pnts[2], color=color, s=scale, alpha=alpha, marker=marker)

def delete_headshape_points(recon_dir=None, subject=None, polhemus_headshape_file=None):
    """Interactively delete headshape points.

    Shows an interactive figure of the polhemus derived headshape points in polhemus space. Points can be clicked on to delete them.

    The figure should be closed upon completion, at which point there is the option to save the deletions.

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories, in the directory structure used by RHINO:
    subject : string
        Subject directory name, in the directory structure used by RHINO.
    polhemus_headshape_file: string
        Full file path to get the polhemus_headshape_file from, and to save any changes to. Note that this is an npy file containing the
        (3 x num_headshapepoints) numpy array of headshape points.
        
    Notes
    -----
    We can call this in two different ways, either:

    1) Specify the subjects_dir AND the subject directory in the 
       directory structure used by RHINO:
    
            delete_headshape_points(recon_dir=recon_dir, subject=subject)
    
    or:
    
    2) Specify the full path to the .npy file containing the (3 x num_headshapepoints) numpy array of headshape points:
    
            delete_headshape_points(polhemus_headshape_file=polhemus_headshape_file)
    """

    if recon_dir is not None and subject is not None:
        coreg_filenames = get_coreg_filenames(recon_dir, subject)
        polhemus_headshape_file = coreg_filenames["polhemus_headshape_file"]
    elif polhemus_headshape_file is not None:
        polhemus_headshape_file = polhemus_headshape_file
        coreg_filenames = {'polhemus_headshape_file': polhemus_headshape_file}
    else:
        ValueError('Invalid inputs. See function\'s documentation.')
      
    polhemus_headshape_polhemus = np.loadtxt(polhemus_headshape_file)

    print("Num headshape points={}".format(polhemus_headshape_polhemus.shape[1]))
    print('Click on points to delete them.')
    print('Press "w" to write changes to the file')
    print('Press "q" to close the figure')
    sys.stdout.flush()

    def scatter_headshapes(ax, x, y, z):
        # Polhemus-derived headshape points
        color, scale, alpha, marker = "red", 8, 0.7, "o"
        ax.scatter(x, y, z, color=color, marker=marker, s=scale, alpha=alpha, picker=5)
        plt.draw()

    x = list(polhemus_headshape_polhemus[0,:])
    y = list(polhemus_headshape_polhemus[1,:])
    z = list(polhemus_headshape_polhemus[2,:])

    # Create scatter plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    scatter_headshapes(ax, x, y, z)

    # Define function to handle click events
    def on_click(event):
        # Get index of clicked point
        ind = event.ind

        # Remove selected points from data arrays
        print('Deleted: {}, {}, {}'.format(x[ind[0]], y[ind[0]], z[ind[0]]))
        sys.stdout.flush()
        
        x.pop(ind[0])
        y.pop(ind[0])
        z.pop(ind[0])

        # Update scatter plot
        ax.cla()
        scatter_headshapes(ax, x, y, z)

    def on_press(event):
        if event.key == 'w':
            polhemus_headshape_polhemus_new = np.array([x, y, z])
            print("Num headshape points remaining={}".format(polhemus_headshape_polhemus_new.shape[1]))
            np.savetxt(coreg_filenames["polhemus_headshape_file"], polhemus_headshape_polhemus_new)
            print('Changes saved to file {}'.format(coreg_filenames["polhemus_headshape_file"]))
        elif event.key == 'q':
            print('Closing figure')
            plt.close(fig)
                    
    # Connect click event to function
    fig.canvas.mpl_connect('pick_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_press)

    plt.show()

def remove_stray_headshape_points(src_dir, subject):
    """Remove stray headshape points.

    Removes headshape points near the nose, on the neck or far away from the head.

    Parameters
    ----------
    src_dir : str
        Path to subjects directory.
    subject : str
        Subject directory name.
    """
    filenames = get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Remove headshape points on the nose
    remove = np.logical_and(hs[1] > max(lpa[1], rpa[1]), hs[2] < nas[2])
    hs = hs[:, ~remove]

    # Remove headshape points on the neck
    remove = hs[2] < min(lpa[2], rpa[2]) - 4
    hs = hs[:, ~remove]

    # Remove headshape points far from the head in any direction
    remove = np.logical_or(hs[0] < lpa[0] - 5, np.logical_or(hs[0] > rpa[0] + 5, hs[1] > nas[1] + 5))
    hs = hs[:, ~remove]

    # Overwrite headshape file
    log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)
