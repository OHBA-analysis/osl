"""Fix structural MRI (sMRI) files.

Replaces the sform with a standard sform.

Note, this script may not be needed. Only use this script if
OSL raises an error regarding the sform code of the sMRIs.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
from pathlib import Path
from shutil import copyfile

import numpy as np
import nibabel as nib

from osl import source_recon

# List of sMRI files we need to fix
smri_files = [
    "sub-001_T1w.nii.gz",
    "sub-002_T1w.nii.gz",
]

# Directory to save fixed sMRIs to
fixed_smri_dir = "data/smri"
os.makedirs(fixed_smri_dir, exist_ok=True)

# Loop through the sMRIs
for input_smri_file in smri_files:

    # Copy the original sMRI file to the output directory
    input_name = Path(input_smri_file).name
    output_smri_file = f"{fixed_smri_dir}/{input_name}"
    print("Saving output to:", output_smri_file)
    copyfile(input_smri_file, output_smri_file)

    # Load the output SMRI file
    smri = nib.load(output_smri_file)

    # Get the original sform header
    sform = smri.header.get_sform()
    sform_std = np.copy(sform)

    # Fix the sform header
    sform_std[0, 0:4] = [-1, 0, 0, 128]
    sform_std[1, 0:4] = [0, 1, 0, -128]
    sform_std[2, 0:4] = [0, 0, 1, -90]
    source_recon.rhino.utils.system_call(
        "fslorient -setsform {} {}".format(
            " ".join(map(str, sform_std.flatten())),
            output_smri_file,
        )
    )

