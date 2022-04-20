#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

from . import preprocessing  # noqa: F401, F403
from . import maxfilter  # noqa: F401, F403
from . import utils  # noqa: F401, F403
from . import report  # noqa: F401, F403
from . import rhino  # noqa: F401, F403
from . import parcellation  # noqa: F401, F403
from . import viz # noqa: F401, F403

# Set logger to only show warning/critical messages
utils.logger.set_up(level='WARNING')
