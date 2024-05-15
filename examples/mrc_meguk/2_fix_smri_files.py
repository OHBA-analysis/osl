"""Fix sform code of structurals.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import numpy as np
import nibabel as nib
from glob import glob

from osl import source_recon

do_oxford = False
do_nottingham = False
do_cardiff = True

raw_dir = "/well/woolrich/projects/mrc_meguk/raw"

def fix_smri(filename):
    smri = nib.load(filename)
    sform = smri.header.get_sform()
    sform_std = np.copy(sform)
    sform_std[0, 0:4] = [-1, 0, 0, 128]
    sform_std[1, 0:4] = [0, 1, 0, -128]
    sform_std[2, 0:4] = [0, 0, 1, -90]
    source_recon.rhino.utils.system_call(
        "fslorient -setsform {} {}".format(
            " ".join(map(str, sform_std.flatten())),
            filename,
        )
    )

if do_oxford:
    smri_dir = "/well/woolrich/projects/mrc_meguk/all_sites/smri/Oxford"
    os.makedirs(smri_dir, exist_ok=True)

    files = sorted(glob(f"{raw_dir}/Oxford/*/anat/*.nii.gz"))
    for file in files:
        if "NeuromagElekta" in file:
            continue
        print("Fixing", file)
        os.system(f"cp {file} {smri_dir}")
        file = f"{smri_dir}/{file.split('/')[-1]}"
        fix_smri(file)

if do_nottingham:
    smri_dir = "/well/woolrich/projects/mrc_meguk/all_sites/smri/Nottingham"
    os.makedirs(smri_dir, exist_ok=True)

    files = sorted(glob(f"{raw_dir}/Nottingham/*/anat/*.nii.gz"))
    for file in files:
        if "CTF" in file:
            continue
        print("Fixing", file)
        os.system(f"cp {file} {smri_dir}")
        file = f"{smri_dir}/{file.split('/')[-1]}"
        fix_smri(file)

if do_cardiff:
    smri_dir = "/well/woolrich/projects/mrc_meguk/all_sites/smri/Cardiff"
    os.makedirs(smri_dir, exist_ok=True)

    files = sorted(glob(f"{raw_dir}/Cardiff/*/anat/*.nii.gz"))
    for file in files:
        if "CTF" in file:
            continue
        print("Fixing", file)
        os.system(f"cp {file} {smri_dir}")
        file = f"{smri_dir}/{file.split('/')[-1]}"
        fix_smri(file)
