"""Tests for passing arguments into batch preprocessing."""

import unittest

import numpy as np

class TestFunctionFinding(unittest.TestCase):

    def test_find_func_in_mne_wrapper(selF):
        from ..preprocessing import find_func
        from ..preprocessing import mne_wrappers as wrappers

        # Check we're finding some common functions
        ff = find_func('notch_filter')
        assert(ff == wrappers.run_mne_notch_filter)

        ff = find_func('resample')
        assert(ff == wrappers.run_mne_resample)

        ff = find_func('pick_channels')
        assert(ff == wrappers.run_mne_pick_channels)

        ff = find_func('pick_types')
        assert(ff == wrappers.run_mne_pick_types)


    def test_find_func_in_mne_object(self):
        import functools
        from ..preprocessing import find_func
        from ..preprocessing import mne_wrappers as wrappers

        # Make sure we have properly set up partial functions based on
        # run_mne_anonymous

        ff = find_func('close')
        assert(isinstance(ff, functools.partial))
        assert(ff.func == wrappers.run_mne_anonymous)
        assert('method' in ff.keywords.keys())
        assert(ff.keywords['method'] == 'close')

        ff = find_func('copy')
        assert(isinstance(ff, functools.partial))
        assert(ff.func == wrappers.run_mne_anonymous)
        assert('method' in ff.keywords.keys())
        assert(ff.keywords['method'] == 'copy')

        ff = find_func('savgol_filter')
        assert(isinstance(ff, functools.partial))
        assert(ff.func == wrappers.run_mne_anonymous)
        assert('method' in ff.keywords.keys())
        assert(ff.keywords['method'] == 'savgol_filter')


    def test_find_func_in_osl_wrapper(self):
        from ..preprocessing import find_func
        from ..preprocessing.osl_wrappers import run_osl_bad_segments, run_osl_bad_channels

        # Check we can find OSL wrapper functions - only 2...
        ff = find_func('bad_segments')
        assert(ff == run_osl_bad_segments)

        ff = find_func('bad_channels')
        assert(ff == run_osl_bad_channels)


    def test_find_func_from_userlist(self):
        from ..preprocessing import find_func
        from ..preprocessing import print_custom_func_info

        # Check that user func is found first
        def filter(x, u):
            return x

        ff = find_func('filter', extra_funcs=[filter])
        assert(ff(1, None) == 1)
