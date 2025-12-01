import warnings
from functools import reduce
from typing import Any, TypedDict, Literal, Required, get_type_hints
from collections.abc import Iterable, Generator

import numpy as np
import snompy
from lmfit import Model, Parameters
from lmfit.models import ConstantModel


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


class Placeholder:
    """Placeholder for multilayer / sample model definition."""

    def make_params(self):
        return Parameters()


class Interface(Placeholder):
    """Define an interface between two sample layers

    By definition, the _next_ layer down is infinite
    """


class FiniteInterface(ConstantModel, Interface):
    """
    Define an finite interface between two sample layers

    Constant value indicates the thickness of the _next_ layer down
    """


class Reference(Placeholder):
    """Define the start of the reference sample"""


def filter_typed_dict(d: dict, td):
    dict_keys = d.keys()
    allowed_keys = get_type_hints(td).keys()
    filtered_keys = [k for k in allowed_keys if k in dict_keys]
    return td(**{k: d[k] for k in filtered_keys})


class SnompyOperationBase:
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        cls.subclasses[cls.__qualname__] = cls

    def __init__(self, parameters):
        self.parameters = parameters

    def __call__(self, sample: snompy.Sample):
        """Override in subclasses."""
        raise NotImplementedError

    @classmethod
    def from_subclass_params(cls, parameters):
        return cls(
            filter_typed_dict(parameters, get_type_hints(cls.__init__)['parameters'])
        )


class EffPolPdmParams(TypedDict, total=False):
    z_tip: float
    r_tip: float
    eps_tip: complex
    alpha_tip: float


class EffPolFdmParams(TypedDict, total=False):
    z_tip: float
    r_tip: float
    L_tip: float
    g_factor: complex
    d_Q0: float  # noqa N815
    d_Q1: float  # noqa N815
    d_Qa: float  # noqa N815
    n_lag: int
    method: Literal["bulk", "multi", "Q_ave"]


class EffPolFdm(SnompyOperationBase):
    def __init__(self, parameters: EffPolFdmParams):
        super().__init__(parameters)

    def __call__(self, sample: snompy.Sample):
        return snompy.fdm.eff_pol(sample=sample, **self.parameters)


class EffPolNFdmParams(EffPolFdmParams, total=False):
    A_tip: Required[float]  # noqa N815
    n: Required[int]
    n_trapz: int


class EffPolNFdm(SnompyOperationBase):
    def __init__(self, parameters: EffPolNFdmParams):
        super().__init__(parameters)

    def __call__(self, sample: snompy.Sample):
        kwargs = self.parameters.copy()
        A_tip = kwargs.pop('A_tip')  # noqa N806
        n = kwargs.pop('n')
        return snompy.fdm.eff_pol_n(sample=sample, A_tip=A_tip, n=n, **kwargs)


class ReflCoefParams(TypedDict, total=False):
    q: float
    theta_in: float
    nu_vac: float
    polarization: Literal["p", "s"]


class SigmaNParams(EffPolNFdmParams, ReflCoefParams, total=False):
    c_r: float


class SigmaN(SnompyOperationBase):
    def __init__(self, parameters: SigmaNParams):
        super().__init__(parameters)

    def __call__(self, sample: snompy.Sample):
        alpha_eff = EffPolNFdm.from_subclass_params(self.parameters)(sample)
        r_coef = sample.refl_coef(**filter_typed_dict(self.parameters, ReflCoefParams))
        c_r = self.parameters['c_r']  # Experimental weighting factor
        return (1 + c_r * r_coef) ** 2 * alpha_eff


def iter_layer(m_iter: Iterable[Model]) -> Generator[Reference | Model, Any, None]:
    """Yield a layer from an interator of models, stopping at Interface"""
    for m in m_iter:
        if isinstance(m, (Interface, Reference)):
            yield m
            break
        yield m


def compose_layer(m_iter: Iterable[Model]) -> Generator[Model, Any, None]:
    """Compose layers from an interator of models, stopping at Reference or end"""
    while True:
        layer = list(iter_layer(m_iter))
        interface = None
        if len(layer) == 0:  # Empty layer
            break
        if isinstance(layer[-1], (Interface, Reference)):
            interface = layer.pop(-1)
        if len(layer) == 0:  # Interface-only layer
            break
        yield reduce(lambda x, y: x + y, layer)
        if isinstance(interface, Interface):
            if isinstance(interface, FiniteInterface):
                yield interface
            continue
        else:
            break


def compose_sample(m_iter: Iterable[Model], op: SnompyOperationBase) -> Model | None:
    sample = list(compose_layer(m_iter))
    if len(sample) == 1:  # Permittivity
        return sample[0]
    elif len(sample) >= 2:
        return MultilayerModel(sample, op)
    else:
        return None


def compose_model(m_list: list[Model], op: SnompyOperationBase) -> Model:
    """"""
    m_iter = iter(m_list)
    sample = compose_sample(m_iter, op)
    reference = compose_sample(m_iter, op)
    if reference is None:
        return sample
    elif sample is None:
        return None
    else:
        return sample / reference


class MultilayerModel(Model):
    """Custom lmfit.Model which combines a list of composite layers into a snompy.Sample

    Evaluates to a function defined by
    """

    def __init__(self, models: list[Model], op, **kws):
        """
        Parameters
        ----------
        left : Model
            Left-hand model.
        right : Model
            Right-hand model.
        op : callable binary operator
            Operator to combine `left` and `right` models.
        **kws : optional
            Additional keywords are passed to `Model` when creating this
            new model.

        Notes
        -----
        The two models can use different independent variables.

        """
        for m in models:
            if not isinstance(m, Model):
                raise ValueError(f'MultilayerModel: argument {m} is not a Model')
        if not callable(op):
            raise ValueError(f'MultilayerModel: operator {op} is not callable')

        self.models = models
        self.op = op

        name_collisions = reduce(
            lambda x, y: x & y, (set(m.param_names) for m in models)
        )
        if len(name_collisions) > 0:
            msg = ''
            for collision in name_collisions:
                msg += (
                    f"\nTwo models have parameters named '{collision}'; "
                    "use distinct names."
                )
            raise NameError(msg)

        # the unique ``independent_vars`` of the left and right model are
        # combined to ``independent_vars`` of the ``CompositeModel``
        if 'independent_vars' not in kws:
            ivars = reduce(lambda x, y: x + y, (m.independent_vars for m in models))
            kws['independent_vars'] = list(np.unique(ivars))
        if 'nan_policy' not in kws:
            kws['nan_policy'] = models[0].nan_policy

        # MultilayerModel cannot have a prefix.
        if 'prefix' in kws:
            warnings.warn("MultilayerModel ignores `prefix` argument", stacklevel=2)
            kws['prefix'] = ''

        def _tmp(self, *args, **kws):
            pass

        Model.__init__(self, _tmp, **kws)
        for m in models:
            prefix = m.prefix
            for basename, hint in m.param_hints.items():
                self.param_hints[f"{prefix}{basename}"] = hint

    def _parse_params(self):
        self._func_haskeywords = reduce(
            lambda x, y: x or y, (m._func_haskeywords for m in self.models)
        )
        self._func_allargs = reduce(
            lambda x, y: x + y, (m._func_allargs for m in self.models)
        )
        self.def_vals = {}
        self.opts = {}
        for m in self.models:
            self.def_vals.update(m.def_vals)
            self.opts.update(m.opts)

    def _reprstring(self, long=True):
        # return (f"({self.left._reprstring(long=long)} "
        #         f"{self._known_ops.get(self.op, self.op)} "
        #         f"{self.right._reprstring(long=long)})")
        # Still needs to show what self.op is
        return tuple(f"MultilayerModel({m._reprstring(long=long)}" for m in self.models)

    def eval(self, params=None, **kwargs):
        """Evaluate model function for composite model."""
        eps_stack = []
        t_stack = []
        for m in self.models:
            eval_m = m.eval(params=params, **kwargs)
            if isinstance(m, FiniteInterface):
                t_stack.append(eval_m)
            else:
                eps_stack.append(eval_m)

        sample = snompy.Sample(eps_stack=eps_stack, t_stack=t_stack, nu_vac=kwargs['x'])

        return self.op(sample=sample)

    def eval_components(self, **kwargs):
        """Return dictionary of name, results for each component."""
        out = {}
        for m in self.models:
            out.update(m.eval_components(**kwargs))
        return out

    def post_fit(self, fitresult):
        """function that is called just after fit, can be overloaded by
        subclasses to add non-fitting 'calculated parameters'
        """
        for m in self.models:
            m.post_fit(fitresult)

    @property
    def param_names(self):
        """Return parameter names for composite model."""
        return reduce(lambda x, y: x + y, (m.param_names for m in self.models))

    @property
    def components(self):
        """Return components for composite model."""
        return reduce(lambda x, y: x + y, (m.components for m in self.models))

    def _get_state(self):
        raise NotImplementedError

    def _set_state(self, state, funcdefs=None):
        raise NotImplementedError

    def _make_all_args(self, params=None, **kwargs):
        """Generate **all** function arguments for all functions."""
        out = {}
        for m in self.models:
            out.update(m._make_all_args(params=params, **kwargs))
        return out
