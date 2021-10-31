import os
import mne
import csv
import glob
import pathlib
import numpy as np


def process_file_inputs(inputs):
    """Process inputs for several cases

    input can be...
    1) string path to unicode file
    2) string path to file or regular-expression matching files
    3) list of string paths to files
    4) list of tuples with path to file and output name pairs
    5) list of MNE objects

    """

    infiles = []
    outnames = []
    check_paths = True

    if isinstance(inputs, str):
        try:
            # Check if path to unicode file...
            open(inputs, 'r')
            infiles, outnames = _load_unicode_inputs(inputs)
        except (UnicodeDecodeError, FileNotFoundError):
            # ...else we have a single path or glob expression
            infiles = glob.glob(inputs)
            outnames = [find_run_id(f) for f in infiles]
    elif isinstance(inputs, (list, tuple)):
        if isinstance(inputs[0], str):
            # We have a list of paths
            infiles = inputs
            outnames = [find_run_id(f) for f in infiles]
        elif isinstance(inputs[0], (list, tuple)):
            # We have a list containing files and output names
            for row in inputs:
                infiles.append(row[0])
                outnames.append(row[1])
        elif isinstance(inputs[0], mne.io.fiff.raw.Raw):
            # We have a list of MNE objects
            infiles = infiles
            check_paths = False

    # Check that files actually exist if we've been passed filenames rather
    # than objects
    good_files = [1 for ii in range(len(infiles))]
    if check_paths:
        for idx, fif in enumerate(infiles):
            if fif.endswith('.ds'):
                good_files[idx] = int(os.path.isdir(fif))
            else:
                good_files[idx] = int(os.path.isfile(fif))
            if good_files[idx] == 0:
                print('File not found: {0}'.format(fif))

    print('{0} files to be processed. {1} good'.format(len(infiles), np.sum(good_files)))

    return infiles, outnames, good_files


def _load_unicode_inputs(fname):
    checked_files = []
    outnames = []
    for row in csv.reader(open(fname, 'r'), delimiter=" "):
        checked_files.append(row[0])
        if len(row) > 1:
            outnames.append(row[1])
        else:
            outnames.append(find_run_id(row[0]))
    return checked_files, outnames


def find_run_id(infile):
    if os.path.split(infile)[1] == 'c,rfDC':
        # We have a BTI scan
        runname = os.path.basename(os.path.dirname(infile))
    elif os.path.splitext(infile)[1] == '.fif':
        # We have a FIF file
        runname = os.path.basename(infile).rstrip('.fif')
    elif os.path.splitext(infile)[1] == '.meg4':
        # We have the meg file from a ds directory
        runname = os.path.basename(infile).rstrip('.ds')
    elif os.path.splitext(infile)[1] == '.ds':
        runname = os.path.basename(infile).rstrip('.ds')
    else:
        raise ValueError('Unable to determine run_id from file {0}'.format(infile))

    return runname


def validate_outdir(outdir):
    """Checks if an output directory exists and if not creates it."""

    outdir = pathlib.Path(outdir)
    if outdir.exists():
        if not os.access(outdir, os.W_OK):
            # Check outdir is a directory
            if not outdir.is_dir():
                raise ValueError("outdir must be the path to a directory.")

            # Check we have write permission
            if not os.access(outdir, os.W_OK):
                raise PermissionError("No write access for {0}".format(outdir))
    else:
        # Output directory doesn't exist
        if outdir.parent.exists():
            # Parent exists, make the output directory
            outdir.mkdir()
        else:
            # Parent doesn't exist
            raise ValueError(
                "Please create the parent directory: {0}".format(outdir.parent)
            )

    return outdir
