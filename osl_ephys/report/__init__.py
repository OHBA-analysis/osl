"""Quality-control reports from preprocessed and source-reconstructed data."""

from .preproc_report import *

import logging

osl_logger = logging.getLogger(__name__)
osl_logger.debug("osl-ephys report init complete")
