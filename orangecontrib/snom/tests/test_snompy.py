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
    FiniteInterface,
    SigmaN,
    EffPolNFdm,
    EffPolFdm,
    EffPolFdmParams,
    EffPolNFdmParams,
    SigmaNParams,
)
from orangecontrib.snom.tests.snompy_examples import (
    snompy_t_dependent_spectra_stepwise,
)


class TestSnompyModel(unittest.TestCase):
    def setUp(self):
        self.snompy_t_dependent_spectra = snompy_t_dependent_spectra_stepwise()
        self.model_list = [
            ConstantModel(name="Air", prefix="const1_"),
            FiniteInterface(prefix="fif2_"),
            LorentzianPermittivityModel(name="PMMA", prefix="lp3_"),
            Interface(),
            ConstantModel(name="Si", prefix="const5_"),
            Reference(),
            ConstantModel(name="Air", prefix="const7_"),
            Interface(),
            DrudePermittivityModel(name="Au", prefix="dp9_"),
        ]
        self.x = np.linspace(1680, 1800, 128) * 1e2  # nuvac
        self.eff_pol_params = EffPolFdmParams(r_tip=30e-9, L_tip=350e-9, method="Q_ave")
        self.eff_pol_n_params = EffPolNFdmParams(
            A_tip=20e-9, n=3, r_tip=30e-9, L_tip=350e-9, method="Q_ave"
        )
        self.sigma_n_params = SigmaNParams(
            **self.eff_pol_n_params, theta_in=np.deg2rad(60), c_r=0.3
        )

    @staticmethod
    def make_params(model):
        # Values from example
        return model.make_params(
            const1_c={'value': 1, 'vary': False},
            fif2_c={'value': 35 * 1e-9, 'vary': False},
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

    def test_fdm_pmma_single(self):
        """Tests the model code generates the same output as the PMMA example, stepwise"""
        alpha = EffPolFdm(self.eff_pol_params)
        alpha_n = EffPolNFdm(self.eff_pol_n_params)
        sigma_n = SigmaN(self.sigma_n_params)
        submodels = {
            "eps_pmma": (self.model_list[2:3], sigma_n),  # op is ignored
            "eps_Au": (self.model_list[8:9], sigma_n),  # op is ignored
            "alpha_eff_pmma": (self.model_list[:5], alpha_n),
            "alpha_eff_pmma_nomod": (self.model_list[:5], alpha),
            "sigma_pmma": (self.model_list[:5], sigma_n),
            "alpha_eff_Au": (self.model_list[6:9], alpha_n),
            "alpha_eff_Au_nomod": (self.model_list[6:9], alpha),
            "sigma_Au": (self.model_list[6:9], sigma_n),
            "eta_n": (self.model_list, sigma_n),  # op(sample) / op(reference)
        }
        for step, snompy in self.snompy_t_dependent_spectra.items():
            if step not in submodels:
                continue
            with self.subTest(msg=f"Testing step {step}"):
                if len(snompy.shape) == 2:
                    # Only compare row from thickness 35 * 1e-9
                    snompy = snompy[-1]
                model = compose_model(*submodels[step])
                model_result = model.fit(
                    np.ones_like(self.x), params=self.make_params(model), x=self.x
                )
                new_eval = np.broadcast_to(model_result.eval(x=self.x), self.x.shape)
                try:
                    np.testing.assert_array_equal(new_eval, snompy)
                except AssertionError:
                    # Plot on failure
                    plot_complex(
                        {'new_model': new_eval, 'snompy': snompy}, self.x, title=step
                    )
                    raise

    def test_fdm_pmma_fit(self):
        """Fit the snompy thickness-dependent PMMA to extract thickness"""
        snompy = self.snompy_t_dependent_spectra["eta_n"][::4]
        t_pmma = self.snompy_t_dependent_spectra['t_pmma'][::4]

        op = SigmaN(self.sigma_n_params)
        model = compose_model(self.model_list, op)
        parameters = self.make_params(model)
        parameters['fif2_c'].set(vary=True)

        t_fit = []

        for s, t in zip(snompy, t_pmma, strict=True):
            model_result = model.fit(s, params=parameters, x=self.x)
            new_eval = np.broadcast_to(model_result.eval(x=self.x), self.x.shape)
            best_t = model_result.best_values['fif2_c']
            tvt = f"t_pmma: {t} t_fit: {best_t} diff: {t - best_t}"
            t_fit.append(t)
            try:
                np.testing.assert_allclose(new_eval, s)
            except AssertionError:
                # Plot on failure
                plot_complex({'new_model': new_eval, 'snompy': s}, self.x, title=tvt)
                raise

        np.testing.assert_allclose(np.asarray(t_fit), t_pmma)

    def test_compose_layer(self):
        """Test the sample / reference compose / split from a model list"""
        model_list = [
            0,
            1,
            2,
            3,
            i34 := FiniteInterface(),
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
        assert sample == [6, i34, 4, 11]  # [0+1+2+3, 4, 5+6]
        assert reference == [45, 54]  # [7+8+9+10+11, 12+13+14+15]


def plot_complex(array_d, nu_vac, title=""):
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
    plt.title(title)
    fig.tight_layout()
    plt.legend()
    plt.show(block=False)
