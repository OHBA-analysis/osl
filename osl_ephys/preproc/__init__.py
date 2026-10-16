"""Sensor-level preprocessing.

The steps of the ``preproc`` section of a config: MNE methods on Raw and
Epochs (``filter``, ``resample``, ``epochs``, ...), the osl-ephys detectors
for bad segments and channels, ICA with automatic or manual labelling, and
the GLM steps. See :mod:`osl_ephys.pipeline` for running a config.
"""

#!/usr/bin/python

from . import mne_wrappers
from . import osl_wrappers

from .batch import *
from .plot_ica import *

# Every wrapper is a step of the preproc section, by its name without the prefix
from osl_ephys.pipeline import registry as _registry

_registry.register_module(mne_wrappers, "preproc")
_registry.register_module(osl_wrappers, "preproc")
