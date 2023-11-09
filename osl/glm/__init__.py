
from .glm_spectrum import *
from .glm_epochs import *

with open(os.path.join(os.path.dirname(__file__), "README.md"), 'r') as f:
    __doc__ = f.read()