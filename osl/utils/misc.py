"""Miscellaneous utility classes and functions.

"""

import logging
import random
import numpy as np


logger = logging.getLogger(__name__)


def set_random_seed(seed=None):
    """Set all random seeds.

    This includes Python's random module and NumPy.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    logger.info(f"Setting random seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)
    return seed