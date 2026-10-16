"""Functions used to find the data files osl-ephys downloads from osl-files."""

import os
import warnings
from os import path

# The names osl-ephys shipped its parcellations under before they moved to
# osl-files. Each was matched to its osl-files file by content (same parcels,
# centroids within 2 mm after regridding), not by name.
OLD_FILENAMES = {
    "Glasser50_space-MNI152NLin6_res-8x8x8.nii.gz": "atlas-Glasser_nparc-50_space-MNI_res-8x8x8.nii.gz",
    "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz": "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz",
    "dk_cortical.nii.gz": "atlas-DK_nparc-68_space-MNI_res-8x8x8.nii.gz",
    "fMRI_parcellation_ds2mm.nii.gz": "atlas-Giles_nparc-38_space-MNI_res-2x2x2.nii.gz",
    "fMRI_parcellation_ds8mm.nii.gz": "atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz": "atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    "fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz": "atlas-Giles_nparc-39_space-MNI_res-8x8x8.nii.gz",
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz": "atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz",
    "aal_cortical_merged_8mm_stacked.nii.gz": "atlas-AAL_nparc-78_space-MNI_res-8x8x8.nii.gz",
    "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_4d_ds8.nii.gz": "atlas-Schaefer_nparc-100_space-MNI_res-8x8x8.nii.gz",
    "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_4d.nii.gz": "atlas-Schaefer_nparc-100_space-MNI_res-2x2x2.nii.gz",
    "fmri_d100_parcellation_with_PCC_tighterMay15_v2_2mm.nii.gz": "atlas-Giles_nparc-39_space-MNI_res-2x2x2.nii.gz",
    "HarvOxf-sub-Schaefer100-combined-2mm_4d.nii.gz": "atlas-SchaeferHarvOxf_nparc-114_space-MNI_res-2x2x2.nii.gz",
    "HarvOxf-sub-Schaefer100-combined-2mm_4d_ds8.nii.gz": "atlas-SchaeferHarvOxf_nparc-114_space-MNI_res-8x8x8.nii.gz",
    "HarvardOxford-sub-prob-bin-2mm.nii.gz": "atlas-HarvOxfSub_nparc-21_space-MNI_res-2x2x2.nii.gz",
    # Kept under their original names in osl-files, for older analyses
    "dk_full.nii.gz": "legacy/dk_full.nii.gz",
    "giles_39_binary.nii.gz": "legacy/giles_39_binary.nii.gz",
    "fmri_d100_parcellation_with_PCC_tighterMay15_v2_6mm_exclusive.nii.gz": "legacy/fmri_d100_parcellation_with_PCC_tighterMay15_v2_6mm_exclusive.nii.gz",
    "fmri_d100_parcellation_with_PCC_reduced_2mm.nii.gz": "legacy/fmri_d100_parcellation_with_PCC_reduced_2mm.nii.gz",
    "WTA_fMRI_parcellation_ds2mm.nii.gz": "legacy/WTA_fMRI_parcellation_ds2mm.nii.gz",
    "WTA_fMRI_parcellation_ds8mm.nii.gz": "legacy/WTA_fMRI_parcellation_ds8mm.nii.gz",
}


def check_exists(filename: str, directory: str) -> str:
    """Looks for a file in the current working directory and in osl-ephys.

    Parameters
    ----------
    filename : str
        Name of file to look for or a path to a file.
    directory : str
        Path to directory to look in.

    Returns
    -------
    filename : str
        Full path to the file if found.

    Raises
    ------
    FileNotFoundError
        If the file could not be found.
    """
    if not path.exists(filename):

        # Mapping from old parcellation filenames to new ones
        if filename in OLD_FILENAMES:
            new = OLD_FILENAMES[filename]
            warnings.warn(
                f"'{filename}' is now '{new}' in osl-files; the old name will "
                "stop working in a future release.",
                FutureWarning,
            )
            filename = new

        if path.exists(f"{directory}/{filename}"):
            filename = f"{directory}/{filename}"
        else:
            raise FileNotFoundError(filename)

    return filename


def _find_package_file(filename):
    from osl_ephys.files import mask, parcellation

    """A parcellation or mask from osl-files, by its name or an old osl-ephys name."""
    for directory in (parcellation.directory, mask.directory):
        try:
            return check_exists(filename, directory)
        except FileNotFoundError:
            continue
    return None


def _find_freesurfer_file(filename):
    from mne import label as mne_label

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
    if not os.path.exists(filename):
        if freesurfer:
            filename = _find_freesurfer_file(filename)
        else:
            filename = _find_package_file(filename)

    if filename is None:
        raise FileNotFoundError(filename)

    return filename
