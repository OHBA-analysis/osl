import matplotlib.pyplot as plt
import mne


def phantom_trigger(phantomraw, savebase=None, *, figsize=(15, 4), dpi=150, transparent=True):
    """ Check the integrity of MEG phantom trigger values.

    Parameters:
        phantomraw (str)  : fif-raw-file name.
        savebase (str)    : Pre-formatted image file name; usage: savebase.format('dipole_trigger').
        figsize (num,num) : Image size in inches.
        dpi (int)         : Image resolution.
        transparent (bool): Transparency flag.

    Return:
        Dictionary (str:str): Status information

    Author:
        Sven Braeutigam, 2021
    """
    rawdat = mne.io.read_raw_fif(phantomraw, verbose=False)
    events = mne.find_events(rawdat, stim_channel='SYS201', consecutive=False, verbose=False)

    diptrg = [0] * 32
    mdptrg = -1
    trgint = [None] * 32
    exttrg = {'init': True, 'count': 0}
    trline = []
    tstamp = -1

    for e in events:
        if e[0] <= tstamp:
            raise Exception('Trigger of order')
        tstamp = e[0]

        trg = e[2]

        if 1 <= trg <= 32:
            diptrg[trg - 1] += 1

            pos = len(trline)
            trline.append(trg)

            if trgint[trg - 1] is None:
                trgint[trg - 1] = [pos, pos]
            else:
                trgint[trg - 1][1] = pos

            if mdptrg < trg:
                mdptrg = trg
            continue

        exttrg['count'] += 1
        if 0 < len(trline):
            exttrg['init'] = False

    figure = plt.figure(figsize=figsize)
    figure.suptitle('Trigger staircase')

    region = figure.add_subplot(1, 1, 1)
    region.scatter(range(0, len(trline)), trline, marker='.')

    xtck = []
    xlbl = []
    for k in range(0, 32):
        i = trgint[k]
        if i is None:
            continue

        xtck.append((i[1] - i[0]) / 2 + i[0])
        xlbl.append(str(k + 1) if k % 2 == 0 else '')
    region.set_xticks(xtck)
    region.set_xticklabels(xlbl)

    region.set_ylim(0, mdptrg + 1)
    region.set_yticks([])

    if savebase is not None:
        plt.savefig(savebase.format('dipole_trigger'), dpi=dpi, transparent=transparent)

    cmin = min(diptrg)
    cmax = max(diptrg)

    return {
        'dipole trigger count': '{} [{}]'.format(
            (cmin, cmax),
            ('OK' if 100 <= cmin and cmax <= 140 else 'MISSING TRIGGER AND/OR TOO MANY REJECTIONS')),
        'non-dipole trigger': '{} [{}]'.format(
            exttrg['count'],
            ('OK' if exttrg['init'] else 'UNEXPECTED/{}'.format(phantomraw)))
    }
