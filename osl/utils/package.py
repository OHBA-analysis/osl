import importlib

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
