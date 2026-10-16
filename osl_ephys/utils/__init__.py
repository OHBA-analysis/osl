#!/usr/bin/python

from . import simulate  #
from . import logger  #
from .study import Study
from .file_handling import *
from .spmio import SPMMEEG
from .simulate import *
from .opm import *
from .package import soft_import, run_package_tests
from .misc import ensure_dir, proc_userargs, mean_psd_in_band
from .version_utils import check_version
from . import run_func

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    __doc__ = f.read()
