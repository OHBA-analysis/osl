"""Coregistration.

"""

import numpy as np
import pandas as pd
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

# Elekta
do_oxford = False
do_cambridge = False

# CTF
do_nottingham = False
do_cardiff = True

raw_dir = "/well/woolrich/projects/mrc_meguk/raw"
preproc_dir = "/well/woolrich/projects/mrc_meguk/all_sites/preproc"
smri_dir = "/well/woolrich/projects/mrc_meguk/all_sites/smri"
coreg_dir = "/well/woolrich/projects/mrc_meguk/all_sites/coreg"

def fix_headshape_points(src_dir, subject, *args, **kwargs):
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Remove headshape points on the nose
    remove = np.logical_and(hs[1] > max(lpa[1], rpa[1]), hs[2] < nas[2])
    hs = hs[:, ~remove]

    # Remove headshape points on the neck
    remove = hs[2] < min(lpa[2], rpa[2]) - 4
    hs = hs[:, ~remove]

    # Remove headshape points far from the head in any direction
    remove = np.logical_or(
        hs[0] < lpa[0] - 5,
        np.logical_or(
            hs[0] > rpa[0] + 5,
            hs[1] > nas[1] + 5,
        ),
    )
    hs = hs[:, ~remove]

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)

def extract_polhemus_from_pos(src_dir, subject, *args, **kwargs):
    """Saves fiducials/headshape from a pos file."""

    # Get coreg filenames
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load pos file
    subject = subject.split("_")[0]
    pos_file = f"{raw_dir}/Nottingham/{subject}/meg/{subject}_headshape.pos"
    utils.logger.log_or_print(f"Saving polhemus from {pos_file}")

    # Load in txt file, these values are in cm in polhemus space
    num_headshape_pnts = int(pd.read_csv(pos_file, header=None).to_numpy()[0])
    data = pd.read_csv(pos_file, header=None, skiprows=[0], delim_whitespace=True)

    # RHINO is going to work with distances in mm
    # So convert to mm from cm, note that these are in polhemus space
    data.iloc[:, 1:4] = data.iloc[:, 1:4] * 10

    # Polhemus fiducial points in polhemus space
    polhemus_nasion = (
        data[data.iloc[:, 0].str.match("nasion")]
        .iloc[0, 1:4].to_numpy().astype("float64").T
    )
    polhemus_rpa = (
        data[data.iloc[:, 0].str.match("right")]
        .iloc[0, 1:4].to_numpy().astype("float64").T
    )
    polhemus_lpa = (
        data[data.iloc[:, 0].str.match("left")]
        .iloc[0, 1:4].to_numpy().astype("float64").T
    )

    # Polhemus headshape points in polhemus space
    polhemus_headshape = (
        data[0:num_headshape_pnts]
        .iloc[:, 1:4].to_numpy().astype("float64").T
    )

    # Save
    np.savetxt(filenames["polhemus_nasion_file"], polhemus_nasion)
    np.savetxt(filenames["polhemus_rpa_file"], polhemus_rpa)
    np.savetxt(filenames["polhemus_lpa_file"], polhemus_lpa)
    np.savetxt(filenames["polhemus_headshape_file"], polhemus_headshape)

def extract_polhemus_from_elc(src_dir, subject, *args, **kwargs):
    """Saves fiducials/headshape from an elc file."""

    # Get coreg filenames
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load elc file
    subject = subject.split("_")[0]
    elc_file = f"{raw_dir}/Cardiff/{subject}/meg/{subject}_headshape.elc"
    utils.logger.log_or_print(f"Saving polhemus from {elc_file}")

    with open(elc_file, "r") as file:
        lines = file.readlines()

        # Polhemus fiducial points in polhemus space
        for i in range(len(lines)):
            if lines[i] == "Positions\n":
                polhemus_nasion = np.array(lines[i + 1].split()[-3:]).astype(np.float64).T
                polhemus_lpa = np.array(lines[i + 2].split()[-3:]).astype(np.float64).T
                polhemus_rpa = np.array(lines[i + 3].split()[-3:]).astype(np.float64).T
                break

        # Polhemus headshape points in polhemus space
        for i in range(len(lines)):
            if lines[i] == "HeadShapePoints\n":
                polhemus_headshape = (
                    np.array([l.split() for l in lines[i + 1:]]).astype(np.float64).T
                )
                break

    # Remove headshape points on the nose
    remove = np.logical_and(
        polhemus_headshape[0] > max(polhemus_lpa[0], polhemus_rpa[0]),
        polhemus_headshape[2] < polhemus_nasion[2],
    )
    polhemus_headshape = polhemus_headshape[:, ~remove]

    # Save
    np.savetxt(filenames["polhemus_nasion_file"], polhemus_nasion)
    np.savetxt(filenames["polhemus_rpa_file"], polhemus_rpa)
    np.savetxt(filenames["polhemus_lpa_file"], polhemus_lpa)
    np.savetxt(filenames["polhemus_headshape_file"], polhemus_headshape)

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    if do_oxford:
        config = """
            source_recon:
            - extract_fiducials_from_fif: {}
            - fix_headshape_points: {}
            - compute_surfaces:
                include_nose: False
            - coregister:
                use_nose: False
                use_headshape: True
        """

        subjects = []
        smri_files = []
        preproc_files = sorted(glob(f"{preproc_dir}/Oxford/*/*_preproc_raw.fif"))
        for file in preproc_files:
            parts = file.split("/")[-2].split("_")
            subjects.append("_".join(parts[:-2]))
            smri_files.append(f"{smri_dir}/Oxford/{parts[0]}_T1w.nii.gz")

        src_dir = f"{coreg_dir}/Oxford"

        source_recon.run_src_batch(
            config,
            src_dir=src_dir,
            subjects=subjects,
            preproc_files=preproc_files,
            smri_files=smri_files,
            extra_funcs=[fix_headshape_points],
            dask_client=True,
        )

    if do_cambridge:
        config = """
            source_recon:
            - extract_fiducials_from_fif: {}
            - fix_headshape_points: {}
            - compute_surfaces:
                include_nose: False
            - coregister:
                use_nose: False
                use_headshape: True
        """

        subjects = []
        smri_files = []
        preproc_files = sorted(glob(f"{preproc_dir}/Cambridge/*/*_preproc_raw.fif"))
        for file in preproc_files:
            parts = file.split("/")[-2].split("_")
            subjects.append("_".join(parts[:-2]))
            smri_files.append(f"{raw_dir}/Cambridge/{parts[0]}/anat/{parts[0]}_T1w.nii.gz")

        src_dir = f"{coreg_dir}/Cambridge"

        source_recon.run_src_batch(
            config,
            src_dir=src_dir,
            subjects=subjects,
            preproc_files=preproc_files,
            smri_files=smri_files,
            extra_funcs=[fix_headshape_points],
            dask_client=True,
        )

    if do_nottingham:
        config = """
            source_recon:
            - extract_polhemus_from_pos: {}
            - fix_headshape_points: {}
            - compute_surfaces:
                include_nose: False
            - coregister:
                use_nose: False
                use_headshape: True
        """

        subjects = []
        smri_files = []
        preproc_files = sorted(glob(f"{preproc_dir}/Nottingham/*/*_preproc_raw.fif"))
        for file in preproc_files:
            parts = file.split("/")[-2].split("_")
            subjects.append("_".join(parts[:-1]))
            smri_files.append(f"{smri_dir}/Nottingham/{parts[0]}_T1w.nii.gz")

        src_dir = f"{coreg_dir}/Nottingham"

        source_recon.run_src_batch(
            config,
            src_dir=src_dir,
            subjects=subjects,
            preproc_files=preproc_files,
            smri_files=smri_files,
            extra_funcs=[fix_headshape_points, extract_polhemus_from_pos],
            dask_client=True,
        )

    if do_cardiff:
        config = """
            source_recon:
            - extract_polhemus_from_elc: {}
            - compute_surfaces:
                include_nose: False
            - coregister:
                use_nose: False
                use_headshape: True
        """

        subjects = []
        smri_files = []
        preproc_files = sorted(glob(f"{preproc_dir}/Cardiff/*/*_preproc_raw.fif"))
        for file in preproc_files:
            parts = file.split("/")[-2].split("_")
            subjects.append("_".join(parts[:-1]))
            smri_files.append(f"{smri_dir}/Cardiff/{parts[0]}_T1w.nii.gz")

        src_dir = f"{coreg_dir}/Cardiff"

        source_recon.run_src_batch(
            config,
            src_dir=src_dir,
            subjects=subjects,
            preproc_files=preproc_files,
            smri_files=smri_files,
            extra_funcs=[fix_headshape_points, extract_polhemus_from_elc],
            dask_client=True,
        )
