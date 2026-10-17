"""Test configuration: no plot windows, in the tests or in the workers they spawn."""

import os

import matplotlib

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")
