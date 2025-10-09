import unittest

import numpy as np
from lmfit.models import ConstantModel

from orangecontrib.snom.model.snompy import (
    compose_model,
    LorentzianPermittivityModel,
    Interface,
    Reference,
    DrudePermittivityModel,
    compose_layer,
)
from orangecontrib.snom.tests.snompy_examples import (
    snompy_t_dependent_spectra_stepwise,
)


class TestSnompyModel(unittest.TestCase):
    def setUp(self):
        self.snompy_t_dependent_spectra = snompy_t_dependent_spectra_stepwise()
        self.model_list = [
            ConstantModel(name="Air", prefix="const1_"),
            Interface(),
            LorentzianPermittivityModel(name="PMMA", prefix="lp3_"),
            Interface(),
            ConstantModel(name="Si", prefix="const5_"),
            Reference(),
            ConstantModel(name="Air", prefix="const7_"),
            Interface(),
            DrudePermittivityModel(name="Au", prefix="dp9_"),
        ]
        self.x = np.linspace(1680, 1800, 128) * 1e2  # nuvac

    @staticmethod
    def make_params(model):
        # Values from example
        return model.make_params(
            const1_c={'value': 1, 'vary': False},
            lp3_nu_j={'value': 1738e2, 'vary': False},
            lp3_A_j={'value': 4.2e8, 'vary': False},
            lp3_gamma_j={'value': 20e2, 'vary': False},
            lp3_eps_inf={'value': 2, 'vary': False},
            const5_c={'value': 11.7, 'vary': False},
            const7_c={'value': 1, 'vary': False},
            dp9_nu_plasma={'value': 7.25e6, 'vary': False},
            dp9_gamma={'value': 2.16e4, 'vary': False},
            dp9_eps_inf={'value': 1, 'vary': False},
        )

    def test_eps_pmma(self):
        snompy = self.snompy_t_dependent_spectra['eps_pmma']
        model = compose_model([self.model_list[2]])
        model_result = model.fit(
            np.ones_like(self.x), params=self.make_params(model), x=self.x
        )
        eval = np.broadcast_to(model_result.eval(x=self.x), self.x.shape)
        test_plot_complex({'new_model': eval, 'snompy': snompy}, self.x)
        np.testing.assert_array_equal(eval, snompy)

    def test_eps_Au(self):
        snompy = self.snompy_t_dependent_spectra['eps_Au']
        model = compose_model([self.model_list[8]])
        model_result = model.fit(
            np.ones_like(self.x), params=self.make_params(model), x=self.x
        )
        eval = np.broadcast_to(model_result.eval(x=self.x), self.x.shape)
        test_plot_complex({'new_model': eval, 'snompy': snompy}, self.x)
        np.testing.assert_array_equal(eval, snompy)

    def test_alpha_eff_pmma_nomod(self):
        snompy = self.snompy_t_dependent_spectra['alpha_eff_pmma_nomod']
        model = compose_model(self.model_list[:5])
        model_result = model.fit(
            np.ones_like(self.x), params=self.make_params(model), x=self.x
        )
        eval = np.broadcast_to(model_result.eval(x=self.x), self.x.shape)
        test_plot_complex({'new_model': eval, 'snompy': snompy}, self.x)
        np.testing.assert_array_equal(eval, snompy)

    def test_alpha_eff_Au_nomod(self):
        snompy = self.snompy_t_dependent_spectra['alpha_eff_Au_nomod']
        model = compose_model(self.model_list[6:9])
        model_result = model.fit(
            np.ones_like(self.x), params=self.make_params(model), x=self.x
        )
        eval = np.broadcast_to(model_result.eval(x=self.x), self.x.shape)
        test_plot_complex({'new_model': eval, 'snompy': snompy}, self.x)
        np.testing.assert_array_equal(eval, snompy)

    # def test_fdm_pmma_single(self):
    #     """Tests the model code generates the same output as the PMMA example"""
    #     snompy_eta_n = snompy_t_dependent_spectra()
    #     model = compose_model(model_list)
    #
    #     v = np.ones_like(x)  # not really fitting, since all parameters are fixed
    #     model_result = model.fit(v, params=parameters, x=x)
    #     eta_n = np.broadcast_to(model_result.eval(x=x), x.shape)
    #     test_plot_complex({'new_model': eta_n, 'snompy': snompy_eta_n}, x)
    #     np.testing.assert_array_equal(eta_n, snompy_eta_n)

    def test_compose_layer(self):
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
        sample = list(compose_layer(m_iter))
        reference = list(compose_layer(m_iter))
        assert sample == [6, 4]  # [0+1+2+3, 4]
        assert reference == [45, 54]  # [7+8+9+10+11, 12+13+14+15]


def test_plot_complex(array_d, nu_vac):
    from matplotlib import pyplot as plt

    # Plot output
    fig, axes = plt.subplots(nrows=2, sharex=True)

    # For neater plotting
    nu_per_cm = nu_vac * 1e-2

    for label, sigma in array_d.items():
        axes[0].plot(nu_per_cm, np.abs(sigma), label=label)
        axes[1].plot(nu_per_cm, np.angle(sigma), label=label)

    axes[0].set_ylabel(r"$s_{" r"}$ / a.u.")
    axes[1].set(
        xlabel=r"$\nu$ / cm$^{-1}$",
        ylabel=r"$\phi_{" r"}$ / radians",
        xlim=(nu_per_cm.max(), nu_per_cm.min()),
    )
    fig.tight_layout()
    plt.legend()
    plt.show(block=False)
