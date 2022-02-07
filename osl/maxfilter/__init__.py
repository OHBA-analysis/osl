#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

from .maxfilter import *  # noqa: F401, F403

import logging
osl_logger = logging.getLogger(__name__)
osl_logger.debug('osl maxfilter init complete')
