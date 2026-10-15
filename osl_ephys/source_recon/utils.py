"""source recon utilities."""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Rob Seymour <rob.seymour@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>

from mne import label as mne_label

import os
import os.path as op

from osl_ephys import files


def _find_package_file(filename):
    """A parcellation or mask from osl-files, by its name or an old osl-ephys name."""
    for directory in (files.parcellation.directory, files.mask.directory):
        try:
            return files.check_exists(filename, directory)
        except FileNotFoundError:
            continue
    # Files not yet moved to osl-files are still shipped with the package
    bundled = op.join(op.dirname(__file__), "files", filename)
    if op.exists(bundled):
        return bundled
    return None


def _find_freesurfer_file(filename):
    avail = mne_label._read_annot_cands(
        os.path.join(os.environ["SUBJECTS_DIR"], "fsaverage", "label")
    )
    if filename in avail:
        filename, hemis = mne_label._get_annot_fname(
            None, "fsaverage", "both", filename, os.environ["SUBJECTS_DIR"]
        )
        return filename


def find_file(filename, freesurfer=False):
    """Look for a file: a path, a name in osl-files, or a FreeSurfer annotation.

    Parameters
    ----------
    filename : str
        Path to file to look for.
    freesurfer : bool, optional
        Should we look in the freesurfer directory?

    Returns
    -------
    filename : str
        Path to file found.
    """
    if not op.exists(filename):
        if freesurfer:
            filename = _find_freesurfer_file(filename)
        else:
            filename = _find_package_file(filename)

    if filename is None:
        raise FileNotFoundError(filename)

    return filename
