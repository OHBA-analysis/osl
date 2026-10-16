#!/usr/bin/python

from . import mne_wrappers  # noqa: F401, F403
from . import osl_wrappers  # noqa: F401, F403

from .batch import *  # noqa: F401, F403
from .plot_ica import *  # noqa: F401, F043

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    __doc__ = f.read()

# Every wrapper is a step of the preproc section, by its name without the prefix
from osl_ephys.pipeline import registry as _registry

_registry.register_module(mne_wrappers, "preproc")
_registry.register_module(osl_wrappers, "preproc")
