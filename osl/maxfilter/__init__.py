#!/usr/bin/python

from .maxfilter import *  # noqa: F401, F403

import logging
osl_logger = logging.getLogger(__name__)
osl_logger.debug('osl maxfilter init complete')

with open(os.path.join(os.path.dirname(__file__), "README.md"), 'r') as f:
    __doc__ = f.read()
