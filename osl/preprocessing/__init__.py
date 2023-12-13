#!/usr/bin/python

from . import mne_wrappers  # noqa: F401, F403
from . import osl_wrappers  # noqa: F401, F403

from .batch import *  # noqa: F401, F403
from .plot_ica import *  # noqa: F401, F043

with open(os.path.join(os.path.dirname(__file__), "README.md"), 'r') as f:
    __doc__ = f.read()