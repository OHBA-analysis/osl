import re
import numpy as np
import matplotlib.pyplot as plt

file_pat = re.compile(r'^File:\s+(\S.+\.fif)\s+IAS:\s+([A-Z]+)\s+SSS:\s+([A-Z]+)\s+tSSS:\s+([A-Z]+)$')
dip_pat = re.compile(r'^(\d\d|\d)\s+\S.+$')
qva_pat = re.compile(r'^\s+Average:\s+\S.+$')
qvr_pat = re.compile(r'\s+All\s+dipole\s+Q-values\s+.+$')


def phantom_report(ascii_report, savebase=None, *, figsize=(15, 4), dpi=150, transparent=True):
    """ Assess and plot MEGIN phantom report.

    Parameters:
        ascii_report (str): report-text-file name.
        savebase (str)    : Pre-formatted image file name; usage: savebase.format('dipole_deviation').
        figsize (num,num) : Image size in inches.
        dpi (int)         : Image resolution.
        transparent (bool): Transparency flag.

    Return:
        Dictionary (str:str): Status information

    Author:
        Sven Braeutigam, 2021
    """
    summary = {}
    count = [0, 0]
    coord = {
        'id': [],
        'dx': [],
        'dy': [],
        'dz': []
    }
    baddip = None

    with open(ascii_report, 'r') as file:
        for line in file.readlines():
            line = line.strip()

            #if m := file_pat.match(line):  # Can include this if we restrict to python >=3.8
            m = file_pat.match(line)
            if m:
                summary['file'] = m.group(1)
                summary['active-shielding'] = 'off' if m.group(2) == 'NO' else 'on'

                maxfilter = 'none'
                if m.group(4) != 'NO':
                    maxfilter = 'tsss'
                elif m.group(3) != 'NO':
                    maxfilter = 'sss'

                summary['maxfilter'] = maxfilter
                continue

            #if m := dip_pat.match(line):  # As above python >=3.8
            m = dip_pat.match(line)
            if m:
                fields = re.split(r'\s+', line)

                count[0] += 1

                if fields[13] == 'OK':
                    coord['id'].append(int(fields[0]))
                    coord['dx'].append(float(fields[9]))
                    coord['dy'].append(float(fields[10]))
                    coord['dz'].append(float(fields[11]))
                    count[1] += 1
                else:
                    baddip = fields[0] if baddip is None else (baddip + ',' + fields[0])
                continue

            #if m := qva_pat.match(line):
            m = qva_pat.match(line)
            if m:
                fields = re.split(r'\s+', line)
                summary['Q-average'] = fields[-1].lower()
                continue

            #if m := qvr_pat.match(line):
            m = qvr_pat.match(line)
            if m:
                fields = re.split(r'\s+', line)
                summary['Q-variance'] = fields[-1].lower()
                continue

    summary['dipole-accuracy'] = \
        '%02d out of %02d dipole(s) within the permissible range' % (count[1], count[0])

    summary['bad-dipoles'] = baddip

    gmin = None
    gmax = None

    scale = lambda x: np.sign(x) * ((int((np.abs(x) + 0.1) * 10)) / 10.0)

    for cr in ['dx', 'dy', 'dz']:
        coord[cr] = np.array(coord[cr], dtype='float64')

        lb = scale(coord[cr].min())
        ub = scale(coord[cr].max())

        if gmin is None or lb < gmin:
            gmin = lb

        if gmax is None or gmax < ub:
            gmax = ub

    figure = plt.figure(figsize=figsize)
    figure.suptitle('Dipole deviation (nominal - actual) [mm]')

    for rc, xl, yl, tl in [(1, 'x', 'y', 'Axial'), (2, 'x', 'z', 'Coronal'), (3, 'y', 'z', 'Sagittal')]:
        region = figure.add_subplot(1, 3, rc)
        region.grid(True)
        region.set_xlim(gmin, gmax)
        region.set_ylim(gmin, gmax)
        region.set_title(tl)
        region.scatter(coord['d' + xl], coord['d' + yl])

    if savebase is not None:
        plt.savefig(savebase.format('dipole_deviation'), dpi=dpi, transparent=transparent)

    return summary
