"""Functions used to find the data files osl-ephys downloads from osl-files."""

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
