from packaging.version import Version, parse
import re
import operator
from importlib.metadata import version

# Housekeeping for logging
import logging
osl_logger = logging.getLogger(__name__)


def _parse_condition(cond):
    """Parse strings defining conditional statements.
    
    Borrowed from EMD package
    """
    name = re.split(r'[=<>!]', cond)[0]
    comp = cond[len(name):]

    if comp[:2] == '==':
        func = operator.eq
    elif comp[:2] == '!=':
        func =  operator.ne
    elif comp[:2] == '<=':
        func =  operator.le
    elif comp[:2] == '>=':
        func =  operator.ge
    elif comp[0] == '<':
        func = operator.lt
    elif comp[0] == '>':
        func = operator.gt
    else:
        print('Comparator not recognised!')

    val = comp.lstrip('!=<>')

    return (name, func, str(val))


def check_version(test_statement, mode='warn'):
    """Check whether the version of a package meets a specified condition.

    Parameters
    ----------
    test_statement : str
        Package version comparison string in the standard format expected by python installs.
        eg 'osl<1.0.0' or 'osl==0.6.dev0'
    mode : {'warn', 'assert'}
        Flag indicating whether to warn the user or raise an error if the comparison fails
    
    """
    test_module, comparator, target_version = _parse_condition(test_statement)

    test_version = Version(version(test_module))
    target_version = Version(target_version)

    if comparator(test_version, target_version) is False:
        msg = "Package '{}' version ({}) fails specified requirement ({})"
        msg = msg.format(test_module, test_version, test_statement)

        if mode == 'warn':
                osl_logger.warning(msg)
        elif mode == 'assert':
                osl_logger.warning(msg)
                raise AssertionError(msg)
