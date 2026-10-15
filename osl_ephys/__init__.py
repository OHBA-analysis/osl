from importlib.metadata import PackageNotFoundError, version

from . import utils  # noqa: F401
from . import preprocessing  # noqa: F401
from . import maxfilter  # noqa: F401
from . import report  # noqa: F401
from . import source_recon  # noqa: F401
from . import glm  # noqa: F401

# The version number comes from the installed package metadata, which
# setuptools_scm derives from the git tag
try:
    __version__ = version("osl-ephys")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
