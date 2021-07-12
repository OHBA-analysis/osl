# This script will be used for the mass-labeling event of the MEGUK partnership data - July 2021
# It intends to speed up the process by loading in the data based on a data name, clearing ica.exclude, plotting the
# components interactively, and saving afterwards - in a researcher specific folder.

def ica_label(dataset):
    global drive, savedir
    import mne
    from osl.preprocessing.osl_plot_ica import plot_ica
    from matplotlib import pyplot as plt
    plt.ion()

    def ica_setup():
        drive = int(input('Do you have access to mark`s (1) or kia`s (2) ohba drive?'))
        if drive==1:
            drive='mwoolrich'
        elif drive==2:
            drive='knobre'
        drive = "".join(('/ohba/pi/', drive, '/datasets/mrc_meguk/'))

        researcher_id = input('What are your initials?').lower()
        savedir = "".join((drive, researcher_id, '/'))
        import os
        if not os.path.isdir(savedir):
            os.mkdir(savedir)

    try: drive
    except NameError: ica_setup()

    print(f'### LABELING DATASET {dataset} ###')
    print('LOADING DATA')
    raw = mne.io.read_raw("".join((drive,dataset, '_raw.fif')))
    ica = mne.preprocessing.read_ica("".join((drive,dataset, '_ica.fif')))
    ica.exclude = [] # empty automatic labeling info
    print(ica.exclude)

    # copy ica to researcher specific folder
    print('COPYING DATA TO RESEARCHER SPECIFIC FOLDER')
    ica.save("".join((savedir, dataset, '_ica.fif')))

    # interactive components plot
    print('INTERACTIVE ICA LABELING')
    plot_ica(ica,raw, block=True)

    print('SAVING DATA')
    ica.save("".join((savedir, dataset, '_ica.fif')))
    del ica, raw
    print(f'### LABELING DATASET {dataset} COMPLETE ###')
