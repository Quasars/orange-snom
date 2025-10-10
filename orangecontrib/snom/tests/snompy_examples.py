import numpy as np
import snompy

# flake8: noqa


def snompy_t_dependent_spectra():
    """snompy/docs/examples/scripts/t_dependent_spectra.py"""
    # fmt: off
    # Set some experimental parameters
    A_tip = 20e-9  # AFM tip tapping amplitude
    r_tip = 30e-9  # AFM tip radius of curvature
    L_tip = 350e-9  # Semi-major axis length of ellipsoid tip model
    n = 3  # Harmonic for demodulation
    theta_in = np.deg2rad(60)  # Light angle of incidence
    c_r = 0.3  # Experimental weighting factor
    nu_vac = np.linspace(1680, 1800, 128) * 1e2  # Vacuum wavenumber
    method = "Q_ave"  # The FDM method to use

    # Semi-infinite superstrate and substrate
    eps_air = 1.0
    eps_Si = 11.7  # Si permitivitty in the mid-infrared

    # Very simplified model of PMMA dielectric function based on ref [1] below
    eps_pmma = snompy.sample.lorentz_perm(
        nu_vac, nu_j=1738e2, gamma_j=20e2, A_j=4.2e8, eps_inf=2
    )
    yield eps_pmma
    t_pmma = np.geomspace(1, 35, 32) * 1e-9  # A range of thicknesses
    yield t_pmma
    sample_pmma = snompy.Sample(
        eps_stack=(eps_air, eps_pmma, eps_Si),
        t_stack=(t_pmma[:, np.newaxis],),
        nu_vac=nu_vac,
    )

    # Model of Au dielectric function from ref [2] below
    eps_Au = snompy.sample.drude_perm(nu_vac, nu_plasma=7.25e6, gamma=2.16e4)
    yield eps_Au
    sample_Au = snompy.bulk_sample(eps_sub=eps_Au, eps_env=eps_air, nu_vac=nu_vac)

    # Measurement
    alpha_eff_pmma = snompy.fdm.eff_pol_n(
        sample=sample_pmma, A_tip=A_tip, n=n, r_tip=r_tip, L_tip=L_tip, method=method
    )
    yield alpha_eff_pmma[-1]
    yield snompy.fdm.eff_pol(
        sample=sample_pmma, r_tip=r_tip, L_tip=L_tip, method=method
    )
    r_coef_pmma = sample_pmma.refl_coef(theta_in=theta_in)
    sigma_pmma = (1 + c_r * r_coef_pmma) ** 2 * alpha_eff_pmma
    yield sigma_pmma

    # Gold reference
    alpha_eff_Au = snompy.fdm.eff_pol_n(
        sample=sample_Au, A_tip=A_tip, n=n, r_tip=r_tip, L_tip=L_tip, method=method
    )
    yield alpha_eff_Au
    yield snompy.fdm.eff_pol(
        sample=sample_Au, r_tip=r_tip, L_tip=L_tip, method=method
    )
    r_coef_Au = sample_Au.refl_coef(theta_in=theta_in)
    sigma_Au = (1 + c_r * r_coef_Au) ** 2 * alpha_eff_Au
    yield sigma_Au

    # Normalised complex scattering
    eta_n = sigma_pmma / sigma_Au
    # fmt: on
    yield eta_n


snompy_t_dependent_spectra_keys = [
    "eps_pmma",
    "t_pmma",
    "eps_Au",
    "alpha_eff_pmma",
    "alpha_eff_pmma_nomod",
    "sigma_pmma",
    "alpha_eff_Au",
    "alpha_eff_Au_nomod",
    "sigma_Au",
    "eta_n",
]


def snompy_t_dependent_spectra_stepwise():
    return {
        k: v
        for k, v in zip(
            snompy_t_dependent_spectra_keys, snompy_t_dependent_spectra(), strict=True
        )
    }
