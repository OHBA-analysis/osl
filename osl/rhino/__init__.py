#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

from . import rhino # noqa: F401, F403
from . import rhino_utils # noqa: F401, F403
from . import examples  # noqa: F401, F403

# Set logger to only show warning/critical messages
utils.logger.set_up(level='WARNING')
