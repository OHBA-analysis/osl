"""Coregisteration.

"""

import numpy as np
import pathlib
from glob import glob
from dask.distributed import Client
from osl import source_recon, utils

BASE_DIR = "/well/woolrich/projects/camcan"
PREPROC_DIR = BASE_DIR + "/winter23/preproc"
COREG_DIR = BASE_DIR + "/winter23/coreg"
ANAT_DIR = BASE_DIR + "/cc700/mri/pipeline/release004/BIDS_20190411/anat"
PREPROC_FILE = PREPROC_DIR + "/mf2pt2_{0}_ses-rest_task-rest_meg/mf2pt2_{0}_ses-rest_task-rest_meg_preproc_raw.fif"
SMRI_FILE = ANAT_DIR + "/{0}/anat/{0}_T1w.nii"
FSL_DIR = "/well/woolrich/projects/software/fsl"

config = """
    source_recon:
    - extract_fiducials_from_fif: {}
    - fix_headshape_points: {}
    - custom_coregister: {}
"""

def fix_headshape_points(src_dir, subject, preproc_file, smri_file, epoch_file):
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Drop nasion by 4cm
    nas[2] -= 40
    distances = np.sqrt(
        (nas[0] - hs[0]) ** 2 + (nas[1] - hs[1]) ** 2 + (nas[2] - hs[2]) ** 2
    )

    # Keep headshape points more than 7 cm away
    keep = distances > 70
    hs = hs[:, keep]

    # Remove anything outside of rpa
    keep = hs[0] < rpa[0]
    hs = hs[:, keep]

    # Remove anything outside of lpa
    keep = hs[0] > lpa[0]
    hs = hs[:, keep]

    if subject in [
        "sub-CC110606", "sub-CC120061", "sub-CC120727", "sub-CC221648", "sub-CC221775",
        "sub-CC121397", "sub-CC121795", "sub-CC210172", "sub-CC210519", "sub-CC210657",
        "sub-CC220419", "sub-CC220974", "sub-CC221107", "sub-CC222956", "sub-CC221828",
        "sub-CC222185", "sub-CC222264", "sub-CC222496", "sub-CC310224", "sub-CC310361",
        "sub-CC310414", "sub-CC312222", "sub-CC320022", "sub-CC320088", "sub-CC320321",
        "sub-CC320336", "sub-CC320342", "sub-CC320448", "sub-CC322186", "sub-CC410243",
        "sub-CC420091", "sub-CC420094", "sub-CC420143", "sub-CC420167", "sub-CC420241",
        "sub-CC420261", "sub-CC420383", "sub-CC420396", "sub-CC420435", "sub-CC420493",
        "sub-CC420566", "sub-CC420582", "sub-CC420720", "sub-CC510255", "sub-CC510321",
        "sub-CC510323", "sub-CC510415", "sub-CC510480", "sub-CC520002", "sub-CC520011",
        "sub-CC520042", "sub-CC520055", "sub-CC520078", "sub-CC520127", "sub-CC520239",
        "sub-CC520254", "sub-CC520279", "sub-CC520377", "sub-CC520391", "sub-CC520477",
        "sub-CC520480", "sub-CC520552", "sub-CC520775", "sub-CC610050", "sub-CC610576",
        "sub-CC620354", "sub-CC620406", "sub-CC620479", "sub-CC620518", "sub-CC620557",
        "sub-CC620572", "sub-CC620610", "sub-CC620659", "sub-CC621011", "sub-CC621128",
        "sub-CC621642", "sub-CC710131", "sub-CC710350", "sub-CC710551", "sub-CC710591",
        "sub-CC710982", "sub-CC711128", "sub-CC720119", "sub-CC720188", "sub-CC720238",
        "sub-CC720304", "sub-CC720358", "sub-CC720511", "sub-CC720622", "sub-CC720685",
        "sub-CC721292", "sub-CC721504", "sub-CC721519", "sub-CC721891", "sub-CC721374",
        "sub-CC722542", "sub-CC722891", "sub-CC121111", "sub-CC121144", "sub-CC210250",
        "sub-CC210422", "sub-CC220519", "sub-CC221209", "sub-CC221487", "sub-CC221595",
        "sub-CC221886", "sub-CC310331", "sub-CC410121", "sub-CC410179", "sub-CC420157",
        "sub-CC510395", "sub-CC610653",
    ]:
        # Remove headshape points 1cm below lpa
        keep = hs[2] > (lpa[2] - 10)
        hs = hs[:, keep]
    elif subject in [
        "sub-CC210023", "sub-CC210124", "sub-CC220132", "sub-CC220203", "sub-CC220223",
        "sub-CC221054", "sub-CC310410", "sub-CC320089", "sub-CC320651", "sub-CC721114",
        "sub-CC320680", "sub-CC320759", "sub-CC320776", "sub-CC320850", "sub-CC320888",
        "sub-CC320904", "sub-CC321073", "sub-CC321154", "sub-CC321174", "sub-CC410084",
        "sub-CC410101", "sub-CC410173", "sub-CC410226", "sub-CC410287", "sub-CC410325",
        "sub-CC410432", "sub-CC420089", "sub-CC420149", "sub-CC420197", "sub-CC420198",
        "sub-CC420217", "sub-CC420222", "sub-CC420260", "sub-CC420324", "sub-CC420356",
        "sub-CC420454", "sub-CC420589", "sub-CC420888", "sub-CC510039", "sub-CC510115",
        "sub-CC510161", "sub-CC510237", "sub-CC510258", "sub-CC510355", "sub-CC510438",
        "sub-CC510551", "sub-CC510609", "sub-CC510629", "sub-CC510648", "sub-CC512003",
        "sub-CC520013", "sub-CC520065", "sub-CC520083", "sub-CC520097", "sub-CC520147",
        "sub-CC520168", "sub-CC520209", "sub-CC520211", "sub-CC520215", "sub-CC520247",
        "sub-CC520395", "sub-CC520398", "sub-CC520503", "sub-CC520584", "sub-CC610052",
        "sub-CC610178", "sub-CC610210", "sub-CC610212", "sub-CC610288", "sub-CC610575",
        "sub-CC610631", "sub-CC620085", "sub-CC620090", "sub-CC620121", "sub-CC620164",
        "sub-CC620444", "sub-CC620496", "sub-CC620526", "sub-CC620592", "sub-CC620793",
        "sub-CC620919", "sub-CC710313", "sub-CC710486", "sub-CC710566", "sub-CC720023",
        "sub-CC720497", "sub-CC720516", "sub-CC720646", "sub-CC721224", "sub-CC721729",
        "sub-CC723395", "sub-CC222326", "sub-CC310160", "sub-CC121479", "sub-CC121685",
        "sub-CC221755", "sub-CC320687", "sub-CC620152", "sub-CC711244",
    ]:
        # Remove headshape points below rpa
        keep = hs[2] > rpa[2]
        hs = hs[:, keep]
    elif subject in [
        "sub-CC210617", "sub-CC220107", "sub-CC220198", "sub-CC220234", "sub-CC220323",
        "sub-CC220335", "sub-CC222125", "sub-CC222258", "sub-CC310008", "sub-CC610046",
        "sub-CC610508",
    ]:
        # Remove headshape points on the face
        keep = np.logical_or(hs[2] > lpa[2], hs[1] < lpa[1])
        hs = hs[:, keep]
    elif subject in [
        "sub-CC410129", "sub-CC410222", "sub-CC410323", "sub-CC410354", "sub-CC420004",
        "sub-CC410390", "sub-CC420348", "sub-CC420623", "sub-CC420729", "sub-CC510043",
        "sub-CC510086", "sub-CC510304", "sub-CC510474", "sub-CC520122", "sub-CC521040",
        "sub-CC610101", "sub-CC610146", "sub-CC610292", "sub-CC620005", "sub-CC620284",
        "sub-CC620413", "sub-CC620490", "sub-CC620515", "sub-CC621199", "sub-CC710037",
        "sub-CC710214", "sub-CC720103", "sub-CC721392", "sub-CC721648", "sub-CC721888",
        "sub-CC722421", "sub-CC722536", "sub-CC720329",
    ]:
        # Remove headshape points 1cm above lpa
        keep = hs[2] > (lpa[2] + 10)
        hs = hs[:, keep]
    elif subject in [
        "sub-CC321428", "sub-CC410097", "sub-CC510076", "sub-CC510220", "sub-CC520560",
        "sub-CC520597", "sub-CC520673", "sub-CC610285", "sub-CC610469", "sub-CC620429",
        "sub-CC620451", "sub-CC620821", "sub-CC710494", "sub-CC722651", "sub-CC110101",
        "sub-CC122172",
    ]:
        # Remove headshape points 2cm above lpa
        keep = hs[2] > (lpa[2] + 20)
        hs = hs[:, keep]
    elif subject in ["sub-CC412004", "sub-CC721704"]:
        # Remove headshape points 3cm above rpa
        keep = hs[2] > (rpa[2] + 30)
        hs = hs[:, keep]
    elif subject in [
        "sub-CC110033", "sub-CC510163", "sub-CC520287", "sub-CC520607", "sub-CC620567",
    ]:
        # Remove headshape points 2cm below lpa
        keep = hs[2] > (lpa[2] - 20)
        hs = hs[:, keep]

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)

def custom_coregister(
    src_dir,
    subject,
    preproc_file,
    smri_file,
    epoch_file,
):
    if subject in [
        "sub-CC110101", "sub-CC110411", "sub-CC120049", "sub-CC420322", "sub-CC120469",
        "sub-CC120208", "sub-CC120309", "sub-CC120319", "sub-CC120347", "sub-CC120376",
        "sub-CC120462", "sub-CC120640", "sub-CC120727", "sub-CC120764", "sub-CC420566",
        "sub-CC122405", "sub-CC410121", "sub-CC410387", "sub-CC122620", "sub-CC210023",
        "sub-CC210148", "sub-CC210172", "sub-CC420888", "sub-CC210519", "sub-CC210617",
        "sub-CC212153", "sub-CC220107", "sub-CC220115", "sub-CC220198", "sub-CC220234",
        "sub-CC220323", "sub-CC220372", "sub-CC220567", "sub-CC220697", "sub-CC220713",
        "sub-CC221002", "sub-CC221031", "sub-CC221033", "sub-CC221040", "sub-CC221107",
        "sub-CC221324", "sub-CC221336", "sub-CC221352", "sub-CC221373", "sub-CC221511",
        "sub-CC221828", "sub-CC221977", "sub-CC222555", "sub-CC510226", "sub-CC222652",
        "sub-CC223085", "sub-CC223115", "sub-CC310008", "sub-CC310051", "sub-CC310052",
        "sub-CC310214", "sub-CC310414", "sub-CC310463", "sub-CC320002", "sub-CC320059",
        "sub-CC510243", "sub-CC320109", "sub-CC320202", "sub-CC320206", "sub-CC320218",
        "sub-CC320325", "sub-CC320379", "sub-CC320417", "sub-CC320429", "sub-CC320500",
        "sub-CC320553", "sub-CC320575", "sub-CC320576", "sub-CC320621", "sub-CC320661",
        "sub-CC320686", "sub-CC320814", "sub-CC321025", "sub-CC321053", "sub-CC321107",
        "sub-CC321137", "sub-CC321281", "sub-CC321291", "sub-CC321331", "sub-CC321368",
        "sub-CC321368", "sub-CC321368", "sub-CC321504", "sub-CC321506", "sub-CC321529",
        "sub-CC321544", "sub-CC321595", "sub-CC321880", "sub-CC321899", "sub-CC410040",
        "sub-CC410091", "sub-CC410113", "sub-CC410129", "sub-CC410177", "sub-CC410289",
        "sub-CC410297", "sub-CC410390", "sub-CC412004", "sub-CC420071", "sub-CC420075",
        "sub-CC420143", "sub-CC420148", "sub-CC420173", "sub-CC420182", "sub-CC420226",
        "sub-CC420324", "sub-CC420623", "sub-CC420776", "sub-CC321431", "sub-CC510392",
        "sub-CC510393", "sub-CC510438", "sub-CC510534", "sub-CC520011", "sub-CC520134",
        "sub-CC520239", "sub-CC520503", "sub-CC520517", "sub-CC520552", "sub-CC520560",
        "sub-CC520673", "sub-CC520868", "sub-CC610028", "sub-CC610076", "sub-CC610594",
        "sub-CC610671", "sub-CC620026", "sub-CC620073", "sub-CC620359", "sub-CC620436",
        "sub-CC620935", "sub-CC621184", "sub-CC621284", "sub-CC710154", "sub-CC711035",
        "sub-CC711158", "sub-CC720400", "sub-CC721291", "sub-CC721377", "sub-CC721434",
        "sub-CC721707", "sub-CC121106", "sub-CC220284", "sub-CC220843", "sub-CC221740",
        "sub-CC720407",
    ]:
        include_nose = True
        use_nose = True
        use_headshape = True
    elif subject in [
        "sub-CC120264", "sub-CC220098", "sub-CC220999", "sub-CC410169", "sub-CC110056",
        "sub-CC110174", "sub-CC112141", "sub-CC120065", "sub-CC210051", "sub-CC221565",
        "sub-CC310203",
    ]:
        include_nose = False
        use_nose = False
        use_headshape = False
    else:
        include_nose = False
        use_nose = False
        use_headshape = True

    if subject in [
        "sub-CC110187", "sub-CC120276", "sub-CC310473", "sub-CC320759", "sub-CC321976",
        "sub-CC410084", "sub-CC410121", "sub-CC420004", "sub-CC420094", "sub-CC420462",
        "sub-CC420589", "sub-CC510050", "sub-CC510076", "sub-CC510208", "sub-CC510321",
        "sub-CC510480", "sub-CC520215", "sub-CC520597", "sub-CC520624", "sub-CC610178",
        "sub-CC610344", "sub-CC610392", "sub-CC610508", "sub-CC620152", "sub-CC620259",
        "sub-CC620284", "sub-CC620685", "sub-CC710088", "sub-CC710429", "sub-CC710664",
        "sub-CC710679", "sub-CC711244", "sub-CC720119", "sub-CC720329",
    ]:
        # These subjects only use a few initialisations which leads to high
        # run-to-run variability. This script may need to be re-run a few times
        # for these subjects to get a good coregistration.
        n_init = 10
    else:
        n_init = 500

    # Compute surfaces
    source_recon.wrappers.compute_surfaces(
        src_dir, subject, preproc_file, smri_file, epoch_file, include_nose
    )

    # Coregister
    source_recon.wrappers.coreg(
        src_dir,
        subject,
        preproc_file,
        smri_file,
        epoch_file,
        use_nose,
        use_headshape,
        n_init=n_init,
    )

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(FSL_DIR)

    # Get subjects
    subjects = []
    for subject in sorted(
        glob(
            PREPROC_DIR
            + "/mf2pt2_*_ses-rest_task-rest_meg"
            + "/mf2pt2_sub-*_ses-rest_task-rest_meg_preproc_raw.fif"
        )
    ):
        subjects.append(pathlib.Path(subject).stem.split("_")[1])

    # Setup files
    smri_files = []
    preproc_files = []
    for subject in subjects:
        smri_files.append(SMRI_FILE.format(subject))
        preproc_files.append(PREPROC_FILE.format(subject))

    # Setup parallel processing
    client = Client(n_workers=16, threads_per_worker=1)

    # Run coregistration
    source_recon.run_src_batch(
        config,
        src_dir=COREG_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_points, custom_coregister],
        dask_client=True,
    )
