#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

# import the logger first - this sets the log-level for the rest of the setup.
from . import logger  ## noqa: F401, F403
from . import simulate  ## noqa: F401, F403
from .studytree import StudyTree  # noqa: F401, F403
from .file_handling import *  # noqa: F401, F403
from .spmio import SPMMEEG  # noqa: F401, F403
from .parallel import initialise_pool  # noqa: F401, F403

import logging
osl_logger = logging.getLogger(__name__)
osl_logger.debug('osl.utils init complete')
