import numpy as np
from AnyQt.QtWidgets import QFormLayout

from Orange.data import Domain

from orangewidget.gui import comboBox

from pySNOM.images import LineLevel

from orangecontrib.spectroscopy.preprocess import SelectColumn, CommonDomain
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange

from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors
from orangecontrib.snom.widgets.preprocessors.utils import PreprocessImageOpts


def get_ndim_hyperspec(data, attrs):
    # mostly copied from orangecontrib.spectroscopy.utils,
    # but returns the indices too
    ndom = Domain(attrs)
    datam = data.transform(ndom)

    from orangecontrib.spectroscopy.utils import axes_to_ndim_linspace

    ls, indices = axes_to_ndim_linspace(datam, attrs)

    # set data
    new_shape = tuple([lsa[2] for lsa in ls]) + (data.X.shape[1],)
    hyperspec = np.ones(new_shape) * np.nan

    hyperspec[indices] = data.X

    return hyperspec, ls, indices


class _LineLevelCommon(CommonDomain):
    def __init__(self, method, domain, image_opts):
        super().__init__(domain)
        self.method = method
        self.image_opts = image_opts

    def transformed(self, data):
        vat = data.domain[self.image_opts["attr_value"]]
        ndom = Domain([vat], data.domain.class_vars, data.domain.metas)
        data = data.transform(ndom)
        xat = data.domain[self.image_opts["attr_x"]]
        yat = data.domain[self.image_opts["attr_y"]]
        hypercube, _, indices = get_ndim_hyperspec(data, (xat, yat))
        transformed = LineLevel(method=self.method).transform(hypercube[:, :, 0])
        out = transformed[indices].reshape(len(data), -1)
        return out


class LineLevelProcessor(PreprocessImageOpts):
    def __init__(self, method="median"):
        self.method = method

    def __call__(self, data, image_opts):
        common = _LineLevelCommon(self.method, data.domain, image_opts)
        at = data.domain[image_opts["attr_value"]].copy(
            compute_value=SelectColumn(0, common)
        )
        domain = Domain([at], data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class LineLevelEditor(BaseEditorOrange):
    name = "Line leveling"
    qualname = "orangecontrib.snom.line_level_test"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.method = 'median'

        form = QFormLayout()
        levelmethod = comboBox(self, self, "method", callback=self.edited.emit)
        levelmethod.addItems(['median', 'mean', 'difference'])
        form.addRow("Leveling method", levelmethod)
        self.controlArea.setLayout(form)

    def activateOptions(self):
        pass  # actions when user starts changing options

    def setParameters(self, params):
        self.levelmethod = params.get("levelmethod", "median")

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        levelmethod = params.get("levelmethod", "median")
        return LineLevelProcessor(method=levelmethod)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(LineLevelEditor, 200)
