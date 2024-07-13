"""
OSL ICA LABEL
Tool for interactive ICA component labeling and rejection. 
Works with command line arguments or as a function call.

"""

# Authors: Mats van Es <mats.vanes@psych.ox.ac.uk>

import sys
import mne
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from osl.preprocessing.plot_ica import plot_ica
from ..utils import logger as osl_logger


def ica_label(preproc_dir, ica_dir, reject=False):
    """Data bookkeeping and wrapping plot_ica.
    
    Parameters
    ----------
    preproc_dir : str
        Path to preprocessed (M/EEG) data.
    ica_dir : str
        Path to ICA data.
    reject : bool or str
        If 'all', reject all components (previously labeled and newly
        labeled). If 'new', reject only newly labeled components. If 
        False (default), only save the ICA data; don't reject any 
        components from the M/EEG data.
    """
    # global drive, savedir
    plt.ion()


    print('LOADING DATA')
    raw = mne.io.read_raw(preproc_dir, preload=True)
    ica = mne.preprocessing.read_ica(ica_dir)
    
    # keep these for later
    if reject=='new':
        exclude_old = deepcopy(ica.exclude)
            
    # interactive components plot
    print('INTERACTIVE ICA LABELING')
    plot_ica(ica, raw, block=True, stop=30)
    plt.pause(0.1)

    if reject == 'all' or reject == 'new':
        print("REMOVING COMPONENTS FROM THE DATA")
        if reject == 'all':
            ica.apply(raw)
        elif reject == 'new':
            # we need to make sure we don't remove components that 
            # were already removed before
            new_ica = deepcopy(ica)
            new_ica.exclude = np.setdiff1d(ica.exclude, exclude_old)
            new_ica.apply(raw)
            
        print("SAVING PREPROCESSED DATA")
        raw.save(preproc_dir, overwrite=True)
    
    print("SAVING ICA DATA")
    ica.save(ica_dir, overwrite=True)
    print(f'LABELING DATASET {preproc_dir.split("/")[-1]} COMPLETE')


def main(argv=None):
    """
    Command-line interface for ica_label.
    
    Parameters
    ----------
    argv : list
        List of strings to be parsed as command-line arguments. If None, 
        sys.argv will be used.
        
    Example
    -------
    From the command line (in the OSL environment), run:
    
    osl_ica_label new /path/to/sub-001_preproc_raw.fif /path/to/sub-001_ica.fif
    
    Then use the GUI to label components (click on the time course to mark, use 
    number keys to label marked components as specific artefacts, and use
    the arrow keys to navigate. Close the plot.
    All/new/none components will be removed from the M/EEG data and saved. The 
    ICA data will be saved with the new labels.
    """

    if argv is None:
        argv = sys.argv[1:]
    
    reject = argv[0]
    if reject == 'False':
        reject = False
        
    preproc_dir = argv[1]
    if len(argv)>2:
        ica_dir = argv[2]
    else:
        ica_dir = preproc_dir.replace('preproc_raw.fif', 'ica.fif')

    print(argv)
    ica_label(preproc_dir, ica_dir, reject)


if __name__ == '__main__':
    main()
