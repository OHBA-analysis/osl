"""
OSL ICA LABEL
Tool for interactive ICA component labeling and rejection. 
Works with command line arguments or as a function call.

"""

# Authors: Mats van Es <mats.vanes@psych.ox.ac.uk>

import sys
import os
import mne
import numpy as np
import pickle
from copy import deepcopy
from matplotlib import pyplot as plt
from osl.preprocessing.plot_ica import plot_ica
from osl.report import plot_bad_ica
from osl.report.preproc_report import gen_html_page, gen_html_summary
from ..utils import logger as osl_logger


def ica_label(preproc_dir, ica_dir, reject=None, report_dir=None):
    """Data bookkeeping and wrapping plot_ica.
    
    Parameters
    ----------
    preproc_dir : str
        Path to preprocessed (M/EEG) data.
    ica_dir : str
        Path to ICA data.
    reject : bool or str
        If 'all', reject all components (previously labeled and newly
        labeled). If 'manual', reject only manually labeled components. If 
        None (default), only save the ICA data; don't reject any 
        components from the M/EEG data.
    """
    # global drive, savedir
    plt.ion()


    print('LOADING DATA')
    raw = mne.io.read_raw(preproc_dir, preload=True)
    ica = mne.preprocessing.read_ica(ica_dir)
    
    # keep these for later
    if reject=='manual':
        exclude_old = deepcopy(ica.exclude)
            
    # interactive components plot
    print('INTERACTIVE ICA LABELING')
    plot_ica(ica, raw, block=True, stop=30)
    plt.pause(0.1)

    if reject == 'all' or reject == 'manual':
        print("REMOVING COMPONENTS FROM THE DATA")
        if reject == 'all':
            ica.apply(raw)
        elif reject == 'manual':
            # we need to make sure we don't remove components that 
            # were already removed before
            new_ica = deepcopy(ica)
            new_ica.exclude = np.setdiff1d(ica.exclude, exclude_old)
            new_ica.apply(raw)
            
        print("SAVING PREPROCESSED DATA")
        raw.save(preproc_dir, overwrite=True)
    
    print("SAVING ICA DATA")
    ica.save(ica_dir, overwrite=True)
    
    if reject is not None:
        print("ATTEMPTING TO UPDATE REPORT")
        try:
            if report_dir is None:
                report_dir = os.path.join("/".join(preproc_dir.split("/")[:-2]), "report")
            
            report_dir_base = deepcopy(report_dir)
            if os.path.exists(os.path.join(report_dir, preproc_dir.split("/")[-2])):
                report_dir = os.path.join(report_dir, preproc_dir.split("/")[-2])
            elif os.path.exists(os.path.join(report_dir, preproc_dir.split("/")[-2]).replace("_raw", "") + "_preproc_raw"):
                report_dir = os.path.join(report_dir, preproc_dir.split("/")[-2]).replace("_raw", "") + "_preproc_raw"
            print(report_dir)
            
            savebase = os.path.join(report_dir, "{0}.png")
            if os.path.exists(os.path.join(report_dir, "ica.png")) or os.path.exists(os.path.join(report_dir, "data.pkl")):
                _ = plot_bad_ica(raw, ica, savebase)

                # try updating the report data
                data = pickle.load(open(os.path.join(report_dir, "data.pkl"), 'rb'))
                if 'plt_ica' not in data.keys():
                    data['plt_ica'] = os.path.join(report_dir.split("/")[-1], "ica.png")

                # update number of bad components
                data['ica_ncomps_rej'] = len(ica.exclude)
                data['ica_ncomps_rej_ecg'] = [len(ica.labels_['ecg']) if 'ecg' in ica.labels_.keys() else 'N/A'][0]
                data['ica_ncomps_rej_eog'] = [len(ica.labels_['eog']) if 'eog' in ica.labels_.keys() else 'N/A'][0]

                # save data
                pickle.dump(data, open(os.path.join(report_dir, "data.pkl"), 'wb'))

                # gen html pages
                gen_html_page(report_dir_base)
                gen_html_summary(report_dir_base)
                
                print("REPORT UPDATED")
        except:
            print("FAILED TO UPDATE REPORT")
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
    From the command line (in the OSL environment), use as follows:

    osl_ica_label reject_argument /path/to/preproc_raw.fif [/path/to/ica.fif] [/path/to/reports_dir]

    The `reject_argument` specifies whether to reject 'all' selected components from the data, only
    the 'manual' rejected, or None (and only save the ICA object, without rejecting components). 
    If the last two optional arguments are not specified, the function will assume their paths from
    the usual OSL structure.
    For example:
    
    osl_ica_label manual /path/to/sub-001_preproc_raw.fif
    
    Then use the GUI to label components (click on the time course to mark, use 
    number keys to label marked components as specific artefacts, and use
    the arrow keys to navigate. Close the plot.
    all/manual/None components will be removed from the M/EEG data and saved. The 
    ICA data will be saved with the new labels. If the report directory is specified
    or in the assumed OSL directory structure, the subject report is updated.

    """

    if argv is None:
        argv = sys.argv[1:]
    
    reject = argv[0]
    if reject == 'None':
        reject = None
        
    preproc_dir = argv[1]
    if len(argv)>2 or (len(argv)==3 and "report" in argv[2]):
        ica_dir = argv[2]
    else:
        ica_dir = preproc_dir.replace('preproc_raw.fif', 'ica.fif')

    if (len(argv)==3 and "report" in argv[2]):
        report_dir = argv[2]
    elif (len(argv)==4 and "report" in argv[3]):
        report_dir = argv[3]
    else:
        # try to find it in the directory structure
        report_dir = None
        
    ica_label(preproc_dir, ica_dir, reject, report_dir)


if __name__ == '__main__':
    main()
