import numpy as np
from AnyQt.QtWidgets import QFormLayout

from orangecontrib.spectroscopy.utils import InvalidAxisException

from orangewidget.gui import comboBox

from pySNOM.images import LineLevel

from orangecontrib.spectroscopy.preprocess import SelectColumn, CommonDomain
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange

from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors
from orangecontrib.snom.preprocess.utils import (
    PreprocessImageOpts,
    get_ndim_hyperspec,
    domain_with_single_attribute_in_x,
)


class _LineLevelCommon(CommonDomain):
    def __init__(self, method, domain, image_opts):
        super().__init__(domain)
        self.method = method
        self.image_opts = image_opts

    def transformed(self, data):
        vat = data.domain[self.image_opts["attr_value"]]
        ndom = domain_with_single_attribute_in_x(vat, data.domain)
        data = data.transform(ndom)
        try:
            hypercube, _, indices = get_ndim_hyperspec(
                data, (self.image_opts["attr_x"], self.image_opts["attr_y"])
            )
            transformed = LineLevel(method=self.method).transform(hypercube[:, :, 0])
            return transformed[indices].reshape(-1, 1)
        except InvalidAxisException:
            return np.full((len(data), 1), np.nan)


class LineLevelProcessor(PreprocessImageOpts):
    def __init__(self, method="median"):
        self.method = method

    def __call__(self, data, image_opts):
        common = _LineLevelCommon(self.method, data.domain, image_opts)
        at = data.domain[image_opts["attr_value"]].copy(
            compute_value=SelectColumn(0, common)
        )
        domain = domain_with_single_attribute_in_x(at, data.domain)
        return data.transform(domain)


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
