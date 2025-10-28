import numpy as np
from lmfit import Parameters
from orangecontrib.spectroscopy.widgets.peakfit_compute import (
    n_best_fit_parameters,
    best_fit_results,
)

from orangecontrib.snom.model.snompy import compose_model
from orangecontrib.snom.widgets.snompy_util import load_list, create_model_list, load_op

lmfit_model = None
lmfit_x = None


def pool_initializer(m_def, x):
    # Pool initializer is used because MultilayerModel is not picklable.
    # Therefore we transfer the preprocessors exported list and build the Model here
    global lmfit_model
    global lmfit_x
    m_def_loaded = load_list(m_def)
    op = load_op(m_def)
    model_list, parameters = create_model_list(m_def_loaded)
    lmfit_model = compose_model(model_list, op), parameters
    assert lmfit_model[0] is not None
    lmfit_x = x


def pool_fit(v):
    x = lmfit_x
    model, parameters = lmfit_model
    model_result = model.fit(v, params=parameters, x=x)
    shape = n_best_fit_parameters(model, parameters)
    bpar = best_fit_results(model_result, x, shape)
    fitted = np.broadcast_to(model_result.eval(x=x), x.shape)

    return (
        {
            'var_names': model_result.var_names,
            'components.prefixes': [c.prefix for c in model_result.components],
        },
        bpar,
        fitted,
        model_result.residual,
    )


def pool_fit2(v, m_def, x) -> Parameters:
    m_def_loaded = load_list(m_def)
    op = load_op(m_def)
    model_list, parameters = create_model_list(m_def_loaded)
    model = compose_model(model_list, op)
    model_result = model.fit(v, params=parameters, x=x)
    return model_result.params
