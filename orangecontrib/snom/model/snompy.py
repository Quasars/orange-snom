from functools import reduce
from typing import Any
from collections.abc import Iterable, Generator

import snompy
from lmfit import Model


# Wrapping existing function, so re-using "A_j" notation (for now).
def lorentz_perm(x, nu_j=0.0, gamma_j=1.0, A_j=1.0, eps_inf=1.0):  # noqa: N803
    """Wraps snompy.sample.lorentz_perm with lmfit-compatible interface."""
    return snompy.sample.lorentz_perm(x, nu_j, gamma_j, A_j=A_j, eps_inf=eps_inf)


class LorentzianPermittivityModel(Model):
    def __init__(
        self,
        independent_vars=['x'],  # noqa: B006 (lmfit compat)
        prefix='',
        nan_policy='raise',
        **kwargs,
    ):
        kwargs.update(
            {
                'prefix': prefix,
                'nan_policy': nan_policy,
                'independent_vars': independent_vars,
            }
        )
        super().__init__(lorentz_perm, **kwargs)


def drude_perm(x, nu_plasma=1.0, gamma=1.0, eps_inf=1.0):
    """Wraps snompy.sample.drude_perm with lmfit-compatible interface.

    drude_perm just wraps the lorentz_perm using the nu_plasma configuration
    and fixed center wavenumber.
    """
    return snompy.sample.drude_perm(x, nu_plasma, gamma, eps_inf=eps_inf)


class DrudePermittivityModel(Model):
    def __init__(
        self,
        independent_vars=['x'],  # noqa: B006 (lmfit compat)
        prefix='',
        nan_policy='raise',
        **kwargs,
    ):
        kwargs.update(
            {
                'prefix': prefix,
                'nan_policy': nan_policy,
                'independent_vars': independent_vars,
            }
        )
        super().__init__(drude_perm, **kwargs)


class Interface:
    """Define an interface between two sample layers"""


class Reference:
    """Define the start of the reference sample"""


def compose_layer(m_iter: Iterable[Model]) -> Generator[Reference | Model, Any, None]:
    """Compose a layer from an interator of models, stopping at Interface"""
    for m in m_iter:
        if isinstance(m, Interface):
            break
        elif isinstance(m, Reference):
            yield m
            break
        yield m


def compose_sample(m_iter: Iterable[Model]) -> Model:
    """Compose a sample from an interator of models, stopping at Reference or end"""
    while True:
        sample = list(compose_layer(m_iter))
        if len(sample) == 0 or any(isinstance(m, Reference) for m in sample):
            break
        yield reduce(lambda x, y: x + y, sample)


def compose_model(m_list: list[Model]) -> Model:
    """"""
    m_iter = iter(m_list)
    sample = list(compose_sample(m_iter))
    reference = list(compose_sample(m_iter))

    return sample, reference
