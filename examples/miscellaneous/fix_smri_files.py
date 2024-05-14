"""Fix sform code of structurals.

This script uses FSL to set the sform code of any structural
whose sform code is not 1 or 4 to make sure it is compatible
with OSL.

Warning: this script will permanently change the SMRI file.
"""

import nibabel as nib

from osl import source_recon

# Paths to files to fix
files = [
    "smri/sub-001.nii.gz",
    "smri/sub-002.nii.gz",
]

for file in files:
    smri = nib.load(file)
    sformcode = smri.header.get_sform(coded=True)[-1]
    if sformcode not in [1, 4]:
        cmd = f"fslorient -setsformcode 1 {file}"
        source_recon.rhino.utils.system_call(cmd)

print("Done")
