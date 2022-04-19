"""Tests for running batch preprocessing - doesn't check that it runs properly,
just that it runs...."""

import unittest

import numpy as np

class TestPreprocessingChain(unittest.TestCase):

    def test_simple_chain(selF):
        from ..preprocessing import find_func
        assert(1 == 1)

