#!/usr/bin/python

from . import mne_wrappers
from . import osl_wrappers

from .batch import *
from .plot_ica import *

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    __doc__ = f.read()

# Every wrapper is a step of the preproc section, by its name without the prefix
from osl_ephys.pipeline import registry as _registry

_registry.register_module(mne_wrappers, "preproc")
_registry.register_module(osl_wrappers, "preproc")
