import warnings
import operator
from functools import reduce, partial
from typing import Any
from collections.abc import Iterable, Generator

import numpy as np
import snompy
from lmfit import Model, CompositeModel, lineshapes


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


def iter_layer(m_iter: Iterable[Model]) -> Generator[Reference | Model, Any, None]:
    """Yield a layer from an interator of models, stopping at Interface"""
    for m in m_iter:
        if isinstance(m, Interface):
            break
        elif isinstance(m, Reference):
            yield m
            break
        yield m


def compose_layer(m_iter: Iterable[Model]) -> Generator[Model, Any, None]:
    """Compose layers from an interator of models, stopping at Reference or end"""
    while True:
        layer = list(iter_layer(m_iter))
        if len(layer) == 0 or any(isinstance(m, Reference) for m in layer):
            break
        yield reduce(lambda x, y: x + y, layer)


def compose_sample(m_iter: Iterable[Model]) -> Model | None:
    sample = list(compose_layer(m_iter))
    if len(sample) == 1:  # Permittivity
        return sample[0]
    elif len(sample) == 2:  # Bulk (single interface)
        return CompositeModel(sample[0], sample[1], eff_pol_with_nuvac)
    elif len(sample) > 2:  # Multilayer
        return MultilayerModel(sample, eff_poll_multi_nuvac)
    else:
        return None


def compose_model(m_list: list[Model]) -> Model:
    """"""
    m_iter = iter(m_list)
    sample = compose_sample(m_iter)
    reference = compose_sample(m_iter)
    if reference is None:
        return sample
    else:
        return sample / reference


def eff_pol(top, bottom, **kwargs):
    eps_stack = (top, bottom)
    nu_vac = kwargs.pop('nu_vac')
    sample = snompy.Sample(eps_stack=eps_stack, nu_vac=nu_vac)
    return snompy.fdm.eff_pol(sample=sample, r_tip=30e-9, L_tip=350e-9, method="Q_ave")


eff_pol_with_nuvac = partial(eff_pol, nu_vac=np.linspace(1680, 1800, 128) * 1e2)


def eff_pol_multi(eps_stack, **kwargs):
    # eps_stack = tuple(eps_stack)
    nu_vac = kwargs.pop('nu_vac')
    # TODO add thickness parameter to somewhere, permittivity?
    t_pmma = 35 * 1e-9
    sample = snompy.Sample(eps_stack=eps_stack, t_stack=(t_pmma,), nu_vac=nu_vac)
    return snompy.fdm.eff_pol(sample=sample, r_tip=30e-9, L_tip=350e-9, method="Q_ave")


eff_poll_multi_nuvac = partial(eff_pol_multi, nu_vac=np.linspace(1680, 1800, 128) * 1e2)


class MultilayerModel(Model):
    """Combine two models (`left` and `right`) with binary operator (`op`).

    Normally, one does not have to explicitly create a `CompositeModel`,
    but can use normal Python operators ``+``, ``-``, ``*``, and ``/`` to
    combine components as in::

    >>> mod = Model(fcn1) + Model(fcn2) * Model(fcn3)

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
        return tuple(f"({m._reprstring(long=long)}" for m in self.models)

    def eval(self, params=None, **kwargs):
        """Evaluate model function for composite model."""
        return self.op([m.eval(params=params, **kwargs) for m in self.models])

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
        return ([m._get_state() for m in self.models], None, self.op.__name__)

    def _set_state(self, state, funcdefs=None):
        return _buildmultilayermodel(state, funcdefs=funcdefs)

    def _make_all_args(self, params=None, **kwargs):
        """Generate **all** function arguments for all functions."""
        out = {}
        for m in self.models:
            out.update(m._make_all_args(params=params, **kwargs))
        return out


def _buildmultilayermodel(state, funcdefs=None):
    """Build Model from saved state.

    Intended for internal use only.

    """
    if len(state) != 3:
        raise ValueError("Cannot restore Model")
    if not isinstance(state[0], list):
        raise ValueError("Cannot restore MultilayerModel")
    known_funcs = {}
    for fname in lineshapes.functions:
        fcn = getattr(lineshapes, fname, None)
        if callable(fcn):
            known_funcs[fname] = fcn
    if funcdefs is None:
        funcdefs = {}
    else:
        known_funcs.update(funcdefs)

    model_states, _, op = state
    if op is None and len(model_states) == 1:
        left = model_states[0]
        if isinstance(left, tuple) and len(left) == 9:
            (fname, func, name, prefix, ivars, pnames, phints, nan_policy, opts) = left
        elif isinstance(left, dict) and 'version' in left:
            # for future-proofing, we could add "if left['version'] == '2':"
            # here to cover cases when 'version' changes
            fname = left.get('funcname', None)
            func = left.get('funcdef', None)
            name = left.get('name', None)
            prefix = left.get('prefix', None)
            ivars = left.get('independent_vars', None)
            pnames = left.get('param_root_names', None)
            phints = left.get('param_hints', None)
            nan_policy = left.get('nan_policy', None)
            opts = left.get('opts', None)
        else:
            raise ValueError("Cannot restore Model: unrecognized state data")

        # if the function definition was passed in, use that!
        if fname in funcdefs and fname != '_eval':
            func = funcdefs[fname]

        if not callable(func) and fname in known_funcs:
            func = known_funcs[fname]

        if func is None:
            raise ValueError("Cannot restore Model: model function not found")

        if fname == '_eval' and isinstance(func, str):
            raise NotImplementedError
            # from .models import ExpressionModel
            # model = ExpressionModel(func, name=name,
            #                         independent_vars=ivars,
            #                         param_names=pnames,
            #                         nan_policy=nan_policy, **opts)

        else:
            model = Model(
                func,
                name=name,
                prefix=prefix,
                independent_vars=ivars,
                param_names=pnames,
                nan_policy=nan_policy,
                **opts,
            )

        for name, hint in phints.items():
            model.set_param_hint(name, **hint)
        return model
    else:
        models = [_buildmultilayermodel(m, funcdefs=funcdefs) for m in model_states]
        return MultilayerModel(models, getattr(operator, op))
