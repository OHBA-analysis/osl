import os
import unittest


class TestModuleStructure(unittest.TestCase):

    def test_module_structure(self):

        try:
            from .. import utils
        except ImportError:
            raise Exception("Unable to import 'utils'")

        try:
            from .. import maxfilter
        except ImportError:
            raise Exception("Unable to import 'maxfilter'")

        try:
            from .. import preprocessing
        except ImportError:
            raise Exception("Unable to import 'preprocessing'")

        try:
            from .. import report
        except ImportError:
            raise Exception("Unable to import 'report'")

        try:
            from .. import source_recon
        except ImportError:
            raise Exception("Unable to import 'source_recon'")


class TestPackageData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.testdir = os.path.dirname(os.path.realpath(__file__))
        cls.osldir = os.path.abspath(os.path.join(cls.testdir, ".."))

    def test_simulation_data(self):
        from .. import files

        assert os.path.exists(files.scanner.neuromag306_template_info)
        for ff in [
            "neuromag306_mvar_params_mag.npy",
            "neuromag306_mvar_residcov_mag.npy",
            "neuromag306_mvar_pcacomp_mag.npy",
            "neuromag306_mvar_params_grad.npy",
            "neuromag306_mvar_residcov_grad.npy",
            "neuromag306_mvar_pcacomp_grad.npy",
        ]:
            assert os.path.exists(files.simulation.file(ff))

    def test_channel_data(self):

        template = os.path.join(self.osldir, "files", "neuromag306_info.yml")
        assert os.path.exists(template)

    def test_parcellation_data(self):
        """Old osl-ephys names resolve to the osl-files parcellations."""
        from ..files.functions import find_file

        for name in [
            "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
            "dk_cortical.nii.gz",
            "fMRI_parcellation_ds8mm.nii.gz",
            "aal_cortical_merged_8mm_stacked.nii.gz",
            "MNI152_T1_8mm_brain.nii.gz",
            "atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
            "HarvOxf-sub-Schaefer100-combined-2mm_4d_ds8.nii.gz",
            "dk_full.nii.gz",
        ]:
            assert os.path.exists(find_file(name)), name
