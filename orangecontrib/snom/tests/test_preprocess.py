import unittest

import numpy as np

from Orange.data import Table

from orangecontrib.spectroscopy.tests.test_preprocess import (
    TestCommonIndpSamplesMixin,
    SMALLER_COLLAGEN,
)

from orangecontrib.snom.preprocess import PhaseUnwrap


class TestPhaseUnwrap(unittest.TestCase, TestCommonIndpSamplesMixin):
    preprocessors = [PhaseUnwrap()]
    data = SMALLER_COLLAGEN

    def test_simple(self):
        data = Table.from_numpy(None, [[1, 1 + 2 * np.pi]])
        f = PhaseUnwrap()
        fdata = f(data)
        # check that unwrap removes jumps greater that 2*pi
        np.testing.assert_array_equal(fdata, [[1, 1]])
