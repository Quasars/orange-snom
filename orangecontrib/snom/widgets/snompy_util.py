from typing import Any

from Orange.data import ContinuousVariable, Domain
from lmfit import Model, Parameters
from orangecontrib.spectroscopy.widgets.owpeakfit import unique_prefix


def load_list(saved) -> list[tuple[Any, dict[str, Any]]]:
    """Load a preprocessor list from a dict."""
    preprocessors = saved.get("preprocessors", [])
    from orangecontrib.snom.widgets.owsnompy import PREPROCESSORS

    qname2ppdef = {ppdef.qualname: ppdef for ppdef in PREPROCESSORS}

    pp_list = []

    # def dropMimeData(data, action, row, column, parent):
    #     if data.hasFormat("application/x-qwidget-ref") and \
    #             action == Qt.CopyAction:
    #         qname = bytes(data.data("application/x-qwidget-ref")).decode()
    #
    #         ppdef = self._qname2ppdef[qname]
    #         item = QStandardItem(ppdef.description.title)
    #         item.setData({}, ParametersRole)
    #         item.setData(ppdef.description.title, Qt.DisplayRole)
    #         item.setData(ppdef, DescriptionRole)
    #         self.preprocessormodel.insertRow(row, [item])
    #         return True
    #     else:
    #         return False
    #
    # model.dropMimeData = dropMimeData

    for qualname, params in preprocessors:
        pp_def = qname2ppdef[qualname]

        # description = pp_def.description
        # item = QStandardItem(description.title)
        # item.setIcon(icon)
        # item.setToolTip(description.summary)
        # item.setData(pp_def, DescriptionRole)
        # item.setData(params, ParametersRole)

        pp_list.append((pp_def, params))
    # TODO do we actually use the qualname after this? might just need params
    return pp_list


def create_model(item, rownum) -> Model:
    desc, params = item
    create = desc.viewclass.createinstance
    prefix = unique_prefix(desc.viewclass, rownum)
    form = params.get('form', None)
    return create(prefix=prefix, form=form)


def prepare_params(item, model) -> Parameters:
    desc, editor_params = item
    translate_hints = desc.viewclass.translate_hints
    all_hints = translate_hints(editor_params)
    for name, hints in all_hints.items():
        model.set_param_hint(name, **hints)
    params = model.make_params()
    return params


def create_model_list(m_def: list[None]) -> tuple[list[Model], Parameters]:
    """create_composite_model() but returns list of models instead"""
    # TODO move to owpeakfit, split create_composite_model up
    n = len(m_def)
    m_list = []
    parameters = Parameters()
    for i in range(n):
        item = m_def[i]
        m = create_model(item, i)
        p = prepare_params(item, m)
        m_list.append(m)
        parameters.update(p)

    return m_list, parameters


def fit_results_table(output, model_result_dict, orig_data):
    """Return best fit parameters as Orange.data.Table"""
    prefixes = model_result_dict['components.prefixes']
    var_names = model_result_dict['var_names']
    features = []
    for prefix in prefixes:
        prefix = prefix.rstrip("_")
        features.append(ContinuousVariable(name=f"{prefix} area"))
        for param in [n for n in var_names if n.startswith(prefix)]:
            features.append(ContinuousVariable(name=param.replace("_", " ")))
    features.append(ContinuousVariable(name="Reduced chi-square"))

    domain = Domain(features, orig_data.domain.class_vars, orig_data.domain.metas)
    out = orig_data.transform(domain)
    with out.unlocked_reference(out.X):
        out.X = output
    return out
