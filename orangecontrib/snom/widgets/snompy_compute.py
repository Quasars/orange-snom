from lmfit import Parameters

from orangecontrib.snom.model.snompy import compose_model
from orangecontrib.snom.widgets.snompy_util import load_list, create_model_list


def pool_fit2(v, m_def, x) -> Parameters:
    m_def = load_list(m_def)
    model_list, parameters = create_model_list(m_def)
    model = compose_model(model_list)
    model_result = model.fit(v, params=parameters, x=x)
    return model_result.params
