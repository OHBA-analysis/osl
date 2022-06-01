#!/usr/bin/python

"""Logging module for OSL

Heavily inspired by logging in OSL.
"""

import yaml
import logging
import logging.config


# Housekeeping for logging
# Add a single null handler until set-up is called, this is activated on import
# to __init__
logging.getLogger('osl').addHandler(logging.NullHandler())

# Initialise logging for this sub-module
logger = logging.getLogger(__name__)

#%% ------------------------------------------------------------


default_config = """
version: 1
loggers:
  osl:
    level: DEBUG
    handlers: [console, file]
    propagate: false

handlers:
  console:
    class : logging.StreamHandler
    formatter: brief
    level   : DEBUG
    stream  : ext://sys.stdout
  file:
    class : logging.handlers.RotatingFileHandler
    formatter: verbose
    filename: {log_file}
    backupCount: 3
    maxBytes: 102400

formatters:
  brief:
    format: '{prefix} %(message)s'
  default:
    format: '[%(asctime)s] {prefix} %(levelname)-8s %(funcName)20s : %(message)s'
    datefmt: '%H:%M:%S'
  verbose:
    format: '[%(asctime)s] {prefix} - %(levelname)s - osl.%(module)s:%(lineno)s - %(funcName)20s() : %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'

disable_existing_loggers: true

"""


def set_up(prefix='', log_file=None, level=None, console_format=None, startup=True):
    """Initialise the OSL module logger.

    Parameters
    ----------
    prefix : str
        Optional prefix to attach to logger output
    log_file : str
        Optional path to a log file to record logger output
    level : {'CRITICAL', 'WARNING', 'INFO', 'DEBUG'}
        String indicating initial logging level
    console_format : str
        Formatting string for console logging.

    """
    # Format config with user options
    if (len(prefix) > 0) and (console_format != 'verbose'):
        prefix = prefix + ' :'
    new_config = default_config.format(prefix=prefix, log_file=log_file)
    # Load config to dict
    new_config = yaml.load(new_config, Loader=yaml.FullLoader)

    # Remove log file from dict if not user requested
    if log_file is None:
        new_config['loggers']['osl']['handlers'] = ['console']
        del new_config['handlers']['file']

    # Configure logger with dict
    logging.config.dictConfig(new_config)

    # Customise options
    if level is not None:
        set_level(level)
    if console_format is not None:
        set_format(formatter=console_format, prefix=prefix)

    if startup:
        # Say hello
        logger.info('OSL Logger Started')

    # Print some info
    if log_file is not None:
        logger.info('logging to file: {0}'.format(log_file))


def set_level(level, handler='console'):
    """Set new logging level for OSL module."""
    logger = logging.getLogger('osl')
    for handler in logger.handlers:
        if handler.get_name() == 'console':
            if level in ['INFO', 'DEBUG']:
                logger.info("OSL logger: handler '{0}' level set to '{1}'".format(handler.get_name(), level))
            handler.setLevel(getattr(logging, level))


def get_level(handler='console'):
    """Return current logging level for OSL module."""
    logger = logging.getLogger('osl')
    for handler in logger.handlers:
        if handler.get_name() == 'console':
            return handler.level

