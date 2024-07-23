import unittest

import numpy as np

from Orange.data import Table

from orangecontrib.snom.preprocess import PhaseUnwrap


class TestPhaseUnwrap(unittest.TestCase):
    def test_simple(self):
        data = Table.from_numpy(None, [[1, 1 + 2 * np.pi]])
        f = PhaseUnwrap()
        fdata = f(data)
        # check that unwrap removes jumps greater that 2*pi
        np.testing.assert_array_equal(fdata, [[1, 1]])
