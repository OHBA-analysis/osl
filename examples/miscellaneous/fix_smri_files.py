"""Fix sform code of structurals.

This script uses FSL to set the sform code of any structural
whose sform code is not 1 or 4 to make sure it is compatible
with OSL.

This script will first create a copy of the sMRI then change
the sform code.
"""

import nibabel as nib

from osl import source_recon

def run(cmd):
    print(cmd)
    source_recon.rhino.utils.system_call(cmd)

# Paths to files to fix
files = [
    "smri/sub-001.nii.gz",
    "smri/sub-002.nii.gz",
]

# Make output directory
output_directory = "data/smri"
run(f"mkdir -p {output_dir}")

for file in files:
    # Copy the original file
    run(f"cp {file} {output_dir}")
    file = f"{output_dir}/{file}"

    # Set the sform code to 1
    run(f"fslorient -setsformcode 1 {file}")

print("Done")
