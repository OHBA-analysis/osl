"""Maxfiltering.

"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import mne
import sys
import ast
import argparse
import tempfile
import numpy as np
from ..utils import validate_outdir, add_subdir, process_file_inputs


parser = argparse.ArgumentParser(description='Batch run Maxfilter on some fif files.')
parser.add_argument('files', type=str,
                    help='plain text file containing full paths to files to be processed')
parser.add_argument('outdir', type=str,
                    help='Path to output directory to save data in')

parser.add_argument('--maxpath', type=str, default='/neuro/bin/util/maxfilter-2.2',
                    help='Path to maxfilter command to use')

parser.add_argument('--mode', type=str, default='standard',
                    help="Running mode for maxfilter. Either 'standard' or 'multistage'")

parser.add_argument('--headpos', action='store_true',
                    help='Output additional head movement parameter file')
parser.add_argument('--movecomp', action='store_true',
                    help='Apply movement compensation')
parser.add_argument('--movecompinter', action='store_true',
                    help='Apply movement compensation on data with intermittent HPI')
parser.add_argument('--nomovecompinter', action='store_true',
                    help='Remove the default movement compensation in the cbu_3stage.')
parser.add_argument('--autobad', action='store_true',
                    help='Apply automatic bad channel detection')
parser.add_argument('--autobad_dur', type=int, default=None,
                    help='Set autobad on with a specific duration')
parser.add_argument('--bad', nargs='+',
                    help='Set specific channels to bad')
parser.add_argument('--badlimit', type=int, default=None,
                    help='Set upper limit for number of bad channels to be removed')
parser.add_argument('--trans', type=str, default=None,
                    help='Transforms the data to the head position in defined file')

parser.add_argument('--origin', nargs='+',
                    help='Set specific sphere origin')
parser.add_argument('--frame', type=str, default=None,
                    help='Set device/dead co-ordinate frame')
parser.add_argument('--force', action='store_true',
                    help='Ignore program warnings')

parser.add_argument('--tsss', action='store_true',
                    help='Apply temporal extension of maxfilter')
parser.add_argument('--st', type=float, default=10,
                    help='Data buffer length for TSSS processing')
parser.add_argument('--corr', type=float, default=0.98,
                    help='Subspace correlation limit for TSSS processing')

parser.add_argument('--inorder', type=int, default=None,
                    help='Set the order of the inside expansion')
parser.add_argument('--outorder', type=int, default=None,
                    help='Set the order of the outside expansion')

parser.add_argument('--hpie', type=int, default=None,
                    help="sets the error limit for hpi coil fitting (def 5 mm)")
parser.add_argument('--hpig', type=float, default=None,
                    help="ets the g-value limit (goodness-of-fit) for hpi coil fitting (def 0.98))")

parser.add_argument('--scanner', type=str, default=None,
                    help="Set CTC and Cal for the OHBA scanner the dataset was collected with \
                          (VectorView, VectorView2 or Neo). This overrides the --ctc and --cal options.")
parser.add_argument('--ctc', type=str, default=None,
                    help='Specify cross-talk calibration file')
parser.add_argument('--cal', type=str, default=None,
                    help='Specify fine-calibration file')

parser.add_argument('--overwrite', action='store_true',
                    help="Overwrite previous output files if they're in the way")
parser.add_argument('--dryrun', action='store_true',
                    help="Don't actually run anything, just print commands that would have been run")


# ----------------------------------------------------------

def _add_headpos(cmd, args, outfif):
    """Estimates and stores head position parameters but does not transform data"""
    if ('headpos' in args) and args['headpos']:
        hp_name = outfif.replace('.fif', '_headpos.log')
        hp_name = os.path.join(args['outdir'], hp_name)
        cmd += ' -hp {0}'.format(hp_name)
    return cmd


def _add_movecomp(cmd, args):
    """Estimates head movements and transforms data.

    Data are transformed to reference head position in continuous raw data"""
    # Add option for Automatic head-movement compensation
    if ('movecomp' in args) and args['movecomp']:
        cmd += ' -movecomp'
    return cmd


def _add_movecompinter(cmd, args):
    """Estimates head movements and transforms data with intermittent HPI

    Data are transformed to reference head position in continuous raw data"""
    # Add option for Automatic head-movement compensation
    if ('movecompinter' in args) and args['movecompinter']:
        cmd += ' -movecomp inter'
    return cmd


def _add_hpie(cmd, args):
    """sets the error limit for hpi coil fitting (def 5 mm)"""
    if ('hpie' in args) and (args['hpie'] is not None):
        cmd += ' -hpie {0}'.format(args['hpie'])
    return cmd


def _add_hpig(cmd, args):
    """sets the g-value limit for hpi coil fitting (def 0.98)"""
    if ('hpig' in args) and (args['hpig'] is not None):
        cmd += ' -hpig {0}'.format(args['hpig'])
    return cmd


def _add_hpisubt(cmd, args):
    """subtracts hpi signals: chpi sine amplitudes, amp + linefreq harmonics,
     or switch off (def = amp)"""
    if ('hpisubt' in args) and (args['hpisubt'] is not None):
        cmd += ' -hpisubt {0}'.format(args['hpisubt'])
    return cmd


def _add_autobad(cmd, args):
    """sets automated bad channel detection on: scan the whole raw data file, off: no autobad"""
    if ('autobad' in args) and args['autobad']:
        cmd += ' -autobad on'
    else:
        cmd += ' -autobad off'
    return cmd


def _add_autobad_dur(cmd, args):
    """sets automated bad channel detection on with specified duration."""
    if ('autobad_dur' in args) and (args['autobad_dur'] is not None):
        cmd += ' -autobad {0}'.format(args['autobad_dur'])
    return cmd


def _add_badlimit(cmd, args):
    """Threshold for bad channel detection (>ave+X*SD)"""
    if ('badlimit' in args) and (args['badlimit'] is not None):
        cmd += ' -badlimit {0}'.format(args['badlimit'])
    return cmd


def _add_bad(cmd, args):
    """sets the list of static bad channels (logical chnos, e.g.: 0323 1042 2631)"""
    if ('bads' in args) and args['bads'] is not None:
        cmd += ' -bad {0}'.format(args['bads'])
    return cmd


def _add_linefreq(cmd, args):
    """sets the basic line interference frequency (50/60Hz)"""
    if ('linefreq' in args) and args['linefreq'] is not None:
        cmd += ' -linefreq {0}'.format(args['linefreq'])
    return cmd


def _add_tsss(cmd, args):
    """Add all tsss related args"""
    # Add option for Temporal Extension of maxfilter
    if ('tsss' in args) and args['tsss']:
        cmd += ' -st {0} -corr {1}'.format(args['st'], args['corr'])
    return cmd


def _add_trans(cmd, args):
    """transforms the data into head position in <fiff_file>"""
    if ('trans' in args) and args['trans'] is not None:
        cmd += ' -trans {0}'.format(args['trans'])
    return cmd


def _add_force(cmd, args):
    """Ignore program warnings...."""
    if ('force' in args) and (args['force'] is not None):
        cmd += ' -force'
    return cmd


def _add_inorder(cmd, args):
    """sets the order of the inside expansion"""
    if ('inorder' in args) and args['inorder'] is not None:
        cmd += ' -in {0}'.format(args['inorder'])
    return cmd


def _add_outorder(cmd, args):
    """sets the order of the outside expansion"""
    if ('outorder' in args) and args['outorder'] is not None:
        cmd += ' -out {0}'.format(args['outorder'])
    return cmd


def _add_ctc(cmd, args):
    """uses the cross-talk matrix in <ctcfile>"""
    if ('ctc' in args) and args['ctc']:
        cmd += ' -ctc {0}'.format(args['ctc'])
    return cmd


def _add_cal(cmd, args):
    """uses the fine-calibration in <calfile>"""
    if ('cal' in args) and args['cal']:
        cmd += ' -cal {0}'.format(args['cal'])
    return cmd


def _add_origin(cmd, args):
    """set a custom sphere origin."""
    if ('origin' in args) and (args['origin'] is not None):
        cmd += ' -origin {0} {1} {2}'.format(*args['origin'])
    return cmd


def _add_frame(cmd, args):
    """set origin frame."""
    if ('frame' in args) and (args['frame'] is not None):
        cmd += ' -frame {0}'.format(args['frame'])
    return cmd


def _add_scanner(cmd, args):
    if ('scanner' in args) is False:
        return cmd
    if args['scanner'] == 'VectorView':
        # Just use defaults
        pass
    elif args['scanner'] == 'VectorView2':
        args['cal'] = '/net/aton/meg_pool/neuromag/databases/sss/sss_cal_3026_171220.dat'
        cmd = _add_cal(cmd, args)
        args['ctc'] = '/net/aton/meg_pool/neuromag/databases/ctc/ct_sparse.fif'
        cmd = _add_ctc(cmd, args)
    elif args['scanner'] == 'Neo':
        #args['cal'] = '/net/aton/meg_pool/data/TriuxNeo/system/sss/sss_cal.dat'
        args['cal'] = '/vols/MEG/TriuxNeo/system/sss/sss_cal.dat'
        cmd = _add_cal(cmd, args)
        #args['ctc'] = '/net/aton/meg_pool/data/TriuxNeo/system/ctc/ct_sparse.fif'
        args['ctc'] = '/vols/MEG/TriuxNeo/system/ctc/ct_sparse.fif'
        cmd = _add_ctc(cmd, args)

    return cmd


def quick_load_dig(fname):
    from mne.io.constants import FIFF
    ff, tree, _ = mne.io.open.fiff_open(fname, preload=False)
    meas = mne.io.tree.dir_tree_find(tree, FIFF.FIFFB_MEAS)
    meas_info = mne.io.tree.dir_tree_find(meas, FIFF.FIFFB_MEAS_INFO)
    dig = mne.io._digitization._read_dig_fif(ff, meas_info)
    return dig


def fit_cbu_origin(infif, outbase=None, remove_nose=True):

    try:
        raw = mne.io.read_raw_fif(infif)
        dig = raw.info['dig']
    except ValueError:
        dig = quick_load_dig(infif)

    # Extract headshape points
    headshape = []
    for dp in dig:
        if dp['kind']._name.find('EXTRA') > 0:
            headshape.append(dp['r'])
    headshape = np.vstack(headshape)

    if remove_nose:
        # Remove nosepoints
        keeps = np.where(np.logical_and(headshape[:, 2] < 0, headshape[:, 1] > 0) == False)[0]  # noqa: E712
        headshape = headshape[keeps, :]

    # Save txt and fit origin
    if outbase is None:
        tmp_hs = tempfile.NamedTemporaryFile(prefix="CBU_MaxfilterOrigin_Headshapes").name
        tmp_fit = tempfile.NamedTemporaryFile(prefix="CBU_MaxfilterOrigin_Fit").name
    else:
        tmp_hs = outbase.format('headshape.txt')
        tmp_fit = outbase.format('headorigin_fit.txt')

    np.savetxt(tmp_hs, headshape)

    cmd = '/neuro/bin/util/fit_sphere_to_points {0} > {1}'.format(tmp_hs, tmp_fit)
    os.system("bash -c '{}'".format(cmd))

    new_origin = np.loadtxt(tmp_fit)[:3] * 1000
    print('fitted origin is {0}'.format(new_origin))

    return new_origin


def run_maxfilter(infif, outfif, args, logfile_tag=''):
    """Wrapper for Elekta Maxfilter.

    Parameters
    ----------
        infif : str
            Path to input fif file (raw data).
        outfif : str
            Path to output fif file (maxfiltered).
        args : dict
            Dictionary of arguments to pass to maxfilter.  See ``help(osl.maxfilter)`` for all options, and 
            Notes for recommendations.
        logfile_tag : str, optional
            Tag to append to logfile name. The default is ''. This is used to
            differentiate between different stages of maxfiltering (e.g., ``'_trans'``, ``'_tsss'``).

    Returns
    -------
        outfif : str
            Path to output fif file (maxfiltered).
        stdlog : str
            Path to logfile.
            
    Notes
    -----
    The recommended use for maxfilter at OHBA is to run multistage maxfiltering, with the following options:
    ``args = {'maxpath': '/neuro/bin/util/maxfilter', 'scanner': 'Neo', 'mode': 'multistage', 'tsss': {}, 'headpos': {}, 'movecomp': {}}``
    """

    basecmd = '{maxpath} -f {infif} -o {outfif}'

    # --------------
    # Format Maxfilter options

    if ('tsss' in args) and args['tsss']:
        outfif = outfif.replace('.fif', 'tsss.fif')
    elif ('trans' in args) and args['trans']:
        outfif = outfif.replace('.fif', 'trans.fif')
    elif logfile_tag != '_trans':
        outfif = outfif.replace('.fif', 'sss.fif')

    # Create base command
    cmd = basecmd.format(maxpath=args['maxpath'], infif=infif, outfif=outfif)

    cmd = _add_headpos(cmd, args, outfif)
    cmd = _add_movecomp(cmd, args)
    cmd = _add_movecompinter(cmd, args)
    cmd = _add_hpie(cmd, args)
    cmd = _add_hpig(cmd, args)
    cmd = _add_hpisubt(cmd, args)
    cmd = _add_linefreq(cmd, args)
    cmd = _add_autobad(cmd, args)
    cmd = _add_autobad_dur(cmd, args)
    cmd = _add_bad(cmd, args)
    cmd = _add_badlimit(cmd, args)
    cmd = _add_force(cmd, args)
    cmd = _add_tsss(cmd, args)
    cmd = _add_origin(cmd, args)
    cmd = _add_frame(cmd, args)
    cmd = _add_trans(cmd, args)
    cmd = _add_inorder(cmd, args)
    cmd = _add_outorder(cmd, args)
    if ('scanner' in args) and args['scanner'] is not None:
        cmd = _add_scanner(cmd, args)
    else:
        cmd = _add_ctc(cmd, args)
        cmd = _add_cal(cmd, args)

    # Add verbose and logfile
    if args['outdir'] == 'adjacent':
        outdir = os.path.split(infif)[0]
    else:
        outdir = args['outdir']
    stdlog = outfif.replace('.fif', '{0}.log'.format(logfile_tag))
    errlog = outfif.replace('.fif', '{0}_err.log'.format(logfile_tag))

    # Set tee to capture both stdout and stderr into separate files
    # https://stackoverflow.com/a/692407
    cmd += ' -v > >(tee -a {stdlog}) 2> >(tee -a {errlog} >&2)'.format(stdlog=stdlog, errlog=errlog)

    # --------------
    # Run Maxfilter

    if args['dryrun']:
        # Dry-run just prints the command
        print(cmd)
        print('\n')
    else:
        print(cmd)
        print('\n')
        # Call maxfilter in a subprocess
        os.system("bash -c '{}'".format(cmd))

    return outfif, stdlog


# -------------------------------------------------

def run_multistage_maxfilter(infif, outbase, args):
    """Wrapper for running :py:func:`run_maxfilter <osl.maxfilter.run_maxfilter>` in three sequential steps:
         
    1. Find Bad Channels
         
    2. Signal Space Separation
         
    3. Translate to reference file
    
    Parameters
    ----------
        infif : str
            Path to input fif file (raw data).
        outbase : str
            output directory.
        args : dict
            Dictionary of arguments to pass to maxfilter. See ``help(osl.maxfilter)`` for all options.

    
    Notes
    -----
    All files are written to disk and the output of each stage is used as the input to the next. 
    
    General advice (from CBU):
    
    * don't use ``'trans'`` with ``'movecomp'``
    
    * don't use ``'autobad'`` with ``'headpos'`` or ``'movecomp'``
    
    * don't use ``'autobad'`` with ``'st'``
    
    References
    ----------
    
    https://imaging.mrc-cbu.cam.ac.uk/meg/Maxfilter
    https://imaging.mrc-cbu.cam.ac.uk/meg/maxbugs
    """

    # --------------------------------------
    # Stage 1 - Find Bad Channels

    outfif = outbase.format('autobad_.fif')
    outlog = outbase.format('autobad.log')

    if os.path.exists(outfif):
        os.remove(outfif)

    # Fixed Args
    stage1_args = {'autobad': True}
    # User args
    for key in ['inorder', 'outorder', 'hpie', 'hpig', 'maxpath', 'origin', 'frame',
                'scanner', 'ctc', 'cal', 'dryrun', 'overwrite', 'outdir']:
        if key in args:
            stage1_args[key] = args[key]

    outfif, outlog = run_maxfilter(infif, outfif, stage1_args)

    if args['dryrun'] is False:
        # Read in bad channels from logfile
        with open(outlog, 'r') as f:
            txt = f.readlines()

        for ii in range(len(txt)):
            if txt[ii][:19] == 'Static bad channels':
                bads = txt[ii].split(': ')[1].split(' ')
                bads = ' '.join([b.strip('\n') for b in bads])
                break
            else:
                bads = None
    else:
        bads = None

    # --------------------------------------
    # Stage 2 - Signal Space Separation

    outfif = outbase.format('.fif')
    outlog = outbase.format('.log')

    if os.path.exists(outfif):
        os.remove(outfif)

    # Fixed Args
    stage2_args = {'autobad': None, 'bads': bads}
    # User args
    for key in ['tsss', 'st', 'corr', 'inorder', 'outorder', 'maxpath', 'origin', 'frame',
                'scanner', 'ctc', 'cal', 'dryrun', 'overwrite', 'hpig', 'hpie',
                'movecomp', 'movecompinter', 'headpos', 'outdir']:
        if key in args:
            stage2_args[key] = args[key]

    outfif, outlog = run_maxfilter(infif, outfif, stage2_args)

    # --------------------------------------
    # Stage 3 - Translate to reference file
    if ('trans' in args) and args['trans'] is not None:

        infif = outfif  # input is output from previous stage
        outfif = outbase.format('.fif')
        outlog = outbase.format('.log')

        if os.path.exists(outfif):
            os.remove(outfif)

        # Fixed Args
        stage3_args = {'autobad': None, 'force': True}
        # User args
        for key in ['maxpath', 'scanner', 'ctc', 'cal',
                    'dryrun', 'overwrite', 'trans', 'outdir']:
            if key in args:
                stage3_args[key] = args[key]

        outfif, outlog = run_maxfilter(infif, outfif, stage3_args)


def run_cbu_3stage_maxfilter(infif, outbase, args):
    """Wrapper for running :py:func:`run_maxfilter <osl.maxfilter.run_maxfilter>` in three 
    sequential steps used by MRC Cognition and Brain Sciences Unit (CBU) in Cambridge:
         
    0. Fit Origin without nose
         
    1. Find Bad Channels
         
    2. Signal Space Separation
    
    3. Translate to default
    
    Parameters
    ----------
        infif : str
            Path to input fif file (raw data).
        outbase : str
            output directory.
        args : dict
            Dictionary of arguments to pass to maxfilter.  See ``help(osl.maxfilter)`` for all options.

    
    Notes
    -----
    All files are written to disk and the output of each stage is used as the input to the next. 

    
    References
    ----------
    
    https://imaging.mrc-cbu.cam.ac.uk/meg/Maxfilter
    https://imaging.mrc-cbu.cam.ac.uk/meg/maxbugs
    """
    
    # --------------------------------------
    # Stage 0 - Fit Origin without nose
    origin = fit_cbu_origin(infif, outbase, remove_nose=True)

    # --------------------------------------
    # Stage 1 - Find Bad Channels

    outfif = outbase.format('autobad_.fif')
    outlog = outbase.format('autobad.log')

    if os.path.exists(outfif):
        os.remove(outfif)

    # Fixed Args
    stage1_args = {'autobad': True, 'origin': origin, 'frame': 'head',
                   'autobad_dur': 1800, 'badlimit': 7,
                   'linefreq': 50, 'hpisubt': 'amp'}
    # User args
    for key in ['inorder', 'outorder', 'hpie', 'hpig', 'maxpath',
                'scanner', 'ctc', 'cal', 'dryrun', 'overwrite', 'outdir']:
        if key in args:
            stage1_args[key] = args[key]

    outfif, outlog = run_maxfilter(infif, outfif, stage1_args)

    if args['dryrun'] is False:
        # Read in bad channels from logfile
        with open(outlog, 'r') as f:
            txt = f.readlines()

        for ii in range(len(txt)):
            if txt[ii][:19] == 'Static bad channels':
                bads = txt[ii].split(': ')[1].split(' ')
                bads = ' '.join([b.strip('\n') for b in bads])
                break
            else:
                bads = None
    else:
        bads = None

    # --------------------------------------
    # Stage 2 - Signal Space Separation

    outfif = outbase.format('.fif')
    outlog = outbase.format('.log')

    if os.path.exists(outfif):
        os.remove(outfif)

    # Fixed Args
    stage2_args = {'autobad': None, 'bads': bads,
                   'origin': origin, 'frame': 'head', 'movecompinter': True,
                   'st': 10, 'corr': 0.98, 'tsss': True,
                   'linefreq': 50, 'hpisubt': 'amp'}
    # User args
    for key in ['inorder', 'outorder', 'maxpath', 'scanner', 'ctc', 'cal',
                'dryrun', 'overwrite', 'hpig', 'hpie', 'headpos', 'outdir']: 
        if key in args:
            stage2_args[key] = args[key]
    
    # movecompinter is allowed be overwritten by user input
    if args['nomovecompinter']:
        stage2_args['movecompinter'] = False
        
    outfif, outlog = run_maxfilter(infif, outfif, stage2_args)

    # --------------------------------------
    # Stage 3 - Translate to default

    infif = outfif  # input is output from previous stage
    outfif = outbase.format('.fif')
    outlog = outbase.format('.log')

    if os.path.exists(outfif):
        os.remove(outfif)

    # Fixed Args
    new_origin = [origin[0], origin[1] - 13, origin[2] + 6]
    stage3_args = {'autobad': None, 'trans': 'default',
                   'origin': list(new_origin), 'frame': 'head', 'force': True}
    # User args
    for key in ['maxpath', 'scanner', 'ctc', 'cal', 'dryrun', 'overwrite', 'outdir']:
        if key in args:
            stage3_args[key] = args[key]

    outfif, outlog = run_maxfilter(infif, outfif, stage3_args)


# -------------------------------------------------

def run_maxfilter_batch(files, outdir, args=None):
    """Batch Maxfiltering.

    Parameters
    ----------
    files : str or list of str
        Path(s) to raw fif files to maxfilter.
    outdir : str
        Path to directory to save output to.
    args : str
        List of additional optional arguments to pass to osl_maxfilter.  See ``help(osl.maxfilter)`` for all options.
        If a string is passed it it split input a list (delimited by spaces).
        E.g. ``args="--maxpath /neuro/bin/util/maxfilter"``
        is equivalent to ``args=["--maxpath", "/neuro/bin/util/maxfilter"]``.
        
    Notes
    -----
    Example use:
    
    
    >>> run_maxfilter_batch(files="/path/to/fif", outdir="/path/to/outdir",
        args="--maxpath /neuro/bin/util/maxfilter --scanner Neo --tsss --mode
        multistage --headpos --movecomp")
    
    """

    if args is None:
        args = []
    if isinstance(args, str):
        args = args.split(" ")
    argv = [files] + [outdir] + args

    args = parser.parse_args(argv)
    args = vars(args)

    if '[' in args['files']:
        args['files'] = ast.literal_eval(args['files'])

    print('\n\nOSL Maxfilter')
    print('-------------\n')
    print(args)
    print()

    infifs, _, _ = process_file_inputs(args['files'])
    good_fifs = [1 for ii in range(len(infifs))]
    for idx, fif in enumerate(infifs):
        if os.path.isfile(fif) is False:
            good_fifs[idx] = 0
            print('File not found: {0}'.format(fif))

    if args['trans'] is not None:
        if os.path.isfile(args['trans']) is False:
            sys.exit('Trans file not found ({0})'.format(args['trans']))

    print('Processing {0} files'.format(sum(good_fifs)))
    if args['outdir'] == 'adjacent':
        print('Outputs will be saved alongside inputs\n\n')
    else:
        if '{' in args['outdir'] and '}' in args['outdir']:
            # validate the parrent outdir - later do so for each subdirectory
            _ = validate_outdir(args['outdir'].split('{')[0])
        else:
            args['outdir'] = validate_outdir(args['outdir'])
        print('Outputs saving to: {0}\n\n'.format(args['outdir']))

    # -------------------------------------------------

    for idx, fif in enumerate(infifs):

        # --------------
        # Format input and output files and run some checks
        print('Processing run {0}/{1} : {2}'.format(idx+1, len(infifs), fif))

        # Skip run if we couldn't find the input file on disk
        if good_fifs[idx] == 0:
            print('Input file not found, skipping run ({0})'.format(fif))
            continue

        # Make an output name : myscan.fif -> myscan_tsss.fif
        outname = os.path.split(fif)[1]

        # Outputfile is output dir + output name
        if args['outdir'] == 'adjacent':
            outfif = fif[:-4]
        else:
            outdir = add_subdir(fif, args['outdir'])
            outdir = validate_outdir(outdir)
            outfif = os.path.join(outdir, outname)[:-4]

        # Skip run if the output exists and we don't want to overwrite
        if os.path.isfile(outfif) and (args['overwrite'] is False):
            print('Existing output found, skipping run ({0})'.format(fif))
            continue

        # Delete previous output if it output exists and we do want to overwrite
        if os.path.isfile(outfif) and args['overwrite']:
            print('Deleting previous output: {0}'.format(outfif))
            os.remove(outfif)

        if args['mode'] == 'standard':
            flag = '_tsss' if args['tsss'] else '_sss'
            outfif = outfif + '_.fif'
            outfif, outlog = run_maxfilter(infifs[idx], outfif, args)
        elif args['mode'] == 'multistage':
            outbase = outfif + '_{0}'
            run_multistage_maxfilter(infifs[idx], outbase, args)
        elif args['mode'] == 'cbu':
            outbase = outfif + '_{0}'
            run_cbu_3stage_maxfilter(infifs[idx], outbase, args)

    print('\nProcessing complete. OHBA-and-out.\n')


# -------------------------------------------------

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    run_maxfilter_batch(argv[0], argv[1], argv[2:])


# ----------------------------------------------------------
# Main user function

if __name__ == '__main__':

    main()
