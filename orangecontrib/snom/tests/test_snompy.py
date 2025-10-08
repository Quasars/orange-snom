import unittest

import numpy as np
from lmfit.models import ConstantModel

from orangecontrib.snom.model.snompy import (
    compose_model,
    LorentzianPermittivityModel,
    Interface,
    Reference,
    DrudePermittivityModel,
    compose_sample,
)
from orangecontrib.snom.tests.snompy_examples import snompy_t_dependent_spectra


class TestSnompyModel(unittest.TestCase):
    def test_fdm_pmma_single(self):
        """Tests the model code generates the same output as the PMMA example"""
        snompy_eta_n = snompy_t_dependent_spectra()
        model_list = [
            ConstantModel(name="Air", prefix="const1_"),
            Interface(),
            LorentzianPermittivityModel(name="PMMA", prefix="lp3_"),
            Interface(),
            ConstantModel(name="Si", prefix="const5_"),
            Reference(),
            ConstantModel(name="Air", prefix="const7_"),
            Interface(),
            DrudePermittivityModel(name="Au", prefix="dp8_"),
        ]
        model = compose_model(model_list)
        # Values from example
        parameters = model.make_params(
            const1_c={'value': 1, 'vary': False},
            lp3_nu_j={'value': 1738e2, 'vary': False},
            lp3_A_j={'value': 4.2e8, 'vary': False},
            lp3_gamma_j={'value': 20e2, 'vary': False},
            lp3_eps_inf={'value': 2, 'vary': False},
            const5_c={'value': 11.7, 'vary': False},
            const7_c={'value': 1, 'vary': False},
            dp9_nu_plasma={'value': 7.25e6, 'vary': False},
            dp9_gamma={'value': 2.14e4, 'vary': False},
        )
        x = np.linspace(1680, 1800, 128) * 1e2  # nuvac
        v = np.ones_like(x)  # not really fitting, since all parameters are fixed
        model_result = model.fit(v, params=parameters, x=x)
        eta_n = np.broadcast_to(model_result.eval(x=x), x.shape)
        np.testing.assert_array_equal(eta_n, snompy_eta_n)

    def test_compose_sample(self):
        """Test the sample / reference compose / split from a model list"""
        model_list = [
            0,
            1,
            2,
            3,
            Interface(),
            4,
            Interface(),
            5,
            6,
            Reference(),
            7,
            8,
            9,
            10,
            11,
            Interface(),
            12,
            13,
            14,
            15,
        ]
        m_iter = iter(model_list)
        sample = list(compose_sample(m_iter))
        reference = list(compose_sample(m_iter))
        assert sample == [6, 4]  # [0+1+2+3, 4]
        assert reference == [45, 54]  # [7+8+9+10+11, 12+13+14+15]
