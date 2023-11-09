from .raw_report import *  # noqa: F401, F403

import logging
osl_logger = logging.getLogger(__name__)
osl_logger.debug('osl report init complete')

with open("osl/report/README.md", 'r') as f:
    __doc__ = f.read()