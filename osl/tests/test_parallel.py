"""Tests for passing arguments into batch preprocessing."""

import unittest

import numpy as np
from dask.distributed import Client, default_client


class TestSimpleDask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        client = Client(n_workers=2, threads_per_worker=1)

    @classmethod
    def tearDownClass(cls):
        client = default_client()
        client.shutdown()

    def test_simple_func(selF):
        from ..utils.parallel import dask_parallel_bag

        def add_five(x):
            return x + 5

        result = dask_parallel_bag(add_five, np.arange(5))
        assert(np.all(result == np.arange(5)+5))

    def test_simple_func_multiple_inputs(selF):
        from ..utils.parallel import dask_parallel_bag

        def multiply(x, y):
            return x * y

        inputs = [(a, a) for a in np.arange(5)]

        result = dask_parallel_bag(multiply, inputs)
        assert(np.all(result == np.array([0, 1, 4, 9, 16])))

    def test_simple_func_with_fixed_args(self):
        from ..utils.parallel import dask_parallel_bag

        def raise_to_power(x, power):
            return x**power

        result = dask_parallel_bag(raise_to_power, np.arange(5),
                                   func_args=[2])
        assert(np.all(result == np.array([0, 1, 4, 9, 16])))

        result = dask_parallel_bag(raise_to_power, np.arange(5),
                                   func_args=[4])
        assert(np.all(result == np.array([0, 1, 16, 81, 256])))

    def test_simple_func_with_fixed_kwargs(self):
        from ..utils.parallel import dask_parallel_bag

        def raise_to_power(x, power=2):
            return x**power

        func_kwargs={'power': 2}
        result = dask_parallel_bag(raise_to_power, np.arange(5),
                                   func_kwargs=func_kwargs)
        assert(np.all(result == np.array([0, 1, 4, 9, 16])))

        func_kwargs={'power': 4}
        result = dask_parallel_bag(raise_to_power, np.arange(5),
                                   func_kwargs=func_kwargs)
        assert(np.all(result == np.array([0, 1, 16, 81, 256])))

    def test_simple_func_with_everything(self):
        from ..utils.parallel import dask_parallel_bag

        def multiply_and_raise_to_power(x, y, const, power=2):
            return (x * y)**power + const

        inputs = [(a, a+2) for a in np.arange(5)]

        func_kwargs={'power': 2}
        result = dask_parallel_bag(multiply_and_raise_to_power, inputs,
                                   func_args=[5],
                                   func_kwargs=func_kwargs)
        assert(np.all(result == np.array([5, 14, 69, 230, 581])))
