import unittest

import numpy as np

from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable

from orangecontrib.spectroscopy.tests.test_preprocess import (
    TestCommonIndpSamplesMixin as TCommonIndpSamplesMixin,  # hide it from pytest
    SMALLER_COLLAGEN,
)

from orangecontrib.snom.preprocess import PhaseUnwrap
from orangecontrib.snom.preprocess.utils import (
    PreprocessImageOpts2DOnlyWhole,
    PreprocessImageOpts2DOnlyWholeReference,
)


class TestPhaseUnwrap(unittest.TestCase, TCommonIndpSamplesMixin):
    preprocessors = [PhaseUnwrap()]
    data = SMALLER_COLLAGEN

    def test_simple(self):
        data = Table.from_numpy(None, [[1, 1 + 2 * np.pi]])
        f = PhaseUnwrap()
        fdata = f(data)
        # check that unwrap removes jumps greater that 2*pi
        np.testing.assert_array_equal(fdata, [[1, 1]])


def test_whitelight_mulcol():
    wl = Table("whitelight.gsf")
    ncol = DiscreteVariable(
        "mask", values=["No", "Yes"], compute_value=lambda d: d.get_column("map_y") > 30
    )
    dom = Domain(
        [ContinuousVariable(name="%0.6f" % i) for i in range(1, 4)],
        wl.domain.class_vars,
        wl.domain.metas + (ncol,),
    )
    out = wl.transform(dom)
    for i in range(2, 4):
        with out.unlocked(out.X):
            out.X[:, i - 1] = out.X[:, 0] * i
    return out


class _MultiplyImage(PreprocessImageOpts2DOnlyWhole):
    def transform_image(self, image, data, mask=None):
        multiplier = float(data.domain.attributes[0].name)
        if mask is None:
            mask = np.full_like(image, True)
        return image * multiplier * mask


class TestPreprocessImageOpts2DOnlyWhole(unittest.TestCase):
    def test_singular(self):
        wl = test_whitelight_mulcol()
        imageopts = {'attr_x': 'map_x', 'attr_y': 'map_y', 'attr_value': '2.000000'}
        proc = _MultiplyImage()
        out = proc(wl, imageopts)
        np.testing.assert_equal(out.X[:, 0], wl.X[:, 1] * 2)

    def test_all(self):
        wl = test_whitelight_mulcol()
        imageopts = {'attr_x': 'map_x', 'attr_y': 'map_y', 'attr_value': '2.000000'}
        proc = _MultiplyImage()
        out = proc(wl, imageopts, run_all=True)
        np.testing.assert_equal(out.X[:, 0], wl.X[:, 0] * 1)
        np.testing.assert_equal(out.X[:, 1], wl.X[:, 0] * 2)
        np.testing.assert_equal(out.X[:, 2], wl.X[:, 0] * 3)

    def test_no_atts(self):
        wl = Table("whitelight.gsf")
        no_atts = wl.transform(Domain([], wl.domain.class_vars, wl.domain.metas))
        imageopts = {'attr_x': 'map_x', 'attr_y': 'map_y', 'attr_value': '1.000000'}
        proc = _MultiplyImage()
        out = proc(no_atts, imageopts, run_all=False)
        np.testing.assert_equal(out.X, no_atts.X)
        out = proc(no_atts, imageopts, run_all=True)
        np.testing.assert_equal(out.X, no_atts.X)

    def test_mask(self):
        wl = test_whitelight_mulcol()
        imageopts = {
            'attr_x': 'map_x',
            'attr_y': 'map_y',
            'attr_value': '2.000000',
            'attr_mask': 'mask',
            'value_mask': 1,
        }
        proc = _MultiplyImage()
        out = proc(wl, imageopts, run_all=True)
        mask = ~(wl.get_column("mask") == 1)
        maski = ~mask
        np.testing.assert_equal(out.X[:, 0][mask], wl.X[:, 0][mask] * 1)
        np.testing.assert_equal(out.X[:, 1][mask], wl.X[:, 0][mask] * 2)
        np.testing.assert_equal(out.X[:, 2][mask], wl.X[:, 0][mask] * 3)
        np.testing.assert_equal(out.X[:, 0][maski], np.nan)
        np.testing.assert_equal(out.X[:, 1][maski], np.nan)
        np.testing.assert_equal(out.X[:, 2][maski], np.nan)


class _MultiplyImageReference(PreprocessImageOpts2DOnlyWholeReference):
    def transform_image(self, image, ref_image, data):
        multiplier = float(data.domain.attributes[0].name)
        return image * ref_image * multiplier


class TestPreprocessImageOpts2DOnlyWholeReference(unittest.TestCase):
    def test_singular(self):
        wl = test_whitelight_mulcol()
        imageopts = {'attr_x': 'map_x', 'attr_y': 'map_y', 'attr_value': '2.000000'}
        proc = _MultiplyImageReference(reference=wl)
        out = proc(wl, imageopts)
        np.testing.assert_equal(out.X[:, 0], wl.X[:, 1] * wl.X[:, 1] * 2)

    def test_all(self):
        wl = test_whitelight_mulcol()
        imageopts = {'attr_x': 'map_x', 'attr_y': 'map_y', 'attr_value': '2.000000'}
        proc = _MultiplyImageReference(reference=wl)
        out = proc(wl, imageopts, run_all=True)
        np.testing.assert_equal(out.X[:, 0], wl.X[:, 0] * wl.X[:, 0] * 1)
        np.testing.assert_equal(out.X[:, 1], wl.X[:, 1] * wl.X[:, 1] * 2)
        np.testing.assert_equal(out.X[:, 2], wl.X[:, 2] * wl.X[:, 2] * 3)

    def test_no_atts(self):
        wl = Table("whitelight.gsf")
        no_atts = wl.transform(Domain([], wl.domain.class_vars, wl.domain.metas))
        imageopts = {'attr_x': 'map_x', 'attr_y': 'map_y', 'attr_value': '1.000000'}
        proc = _MultiplyImageReference(reference=wl)
        out = proc(no_atts, imageopts, run_all=False)
        np.testing.assert_equal(out.X, no_atts.X)
        out = proc(no_atts, imageopts, run_all=True)
        np.testing.assert_equal(out.X, no_atts.X)
