import importlib
import os

# Housekeeping for logging
import logging
logger = logging.getLogger(__name__)

def soft_import(package):
    """Try to import a package raising friendly error if not present."""
    try:
        module = importlib.import_module(package)
    except (ImportError, ModuleNotFoundError):
        msg = f"Package '{package}' is required for this "
        msg += "function to run but cannot be imported. "
        msg += "Please install it into your python environment to continue."
        raise ModuleNotFoundError(msg)

    return module


def run_package_tests():
    """Run OSL tests from within python

    https://docs.pytest.org/en/7.1.x/how-to/usage.html

    Notes
    -----
    Calling pytest.main() will result in importing your tests and any modules
    that they import. Due to the caching mechanism of python’s import system,
    making subsequent calls to pytest.main() from the same process will not
    reflect changes to those files between the calls. For this reason, making
    multiple calls to pytest.main() from the same process
    (in order to re-run tests, for example) is not recommended.

    """
    import pytest

    thisdir = os.path.dirname(os.path.realpath(__file__))
    installdir = os.path.abspath(os.path.join(thisdir, '..'))
    canarypth = os.path.join(installdir, 'tests', 'test_00_package_canary.py')
    print(installdir)

    out = pytest.main(['-x', canarypth])

