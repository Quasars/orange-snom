# this is just an example of registration

from AnyQt.QtWidgets import QFormLayout

from Orange.data import Domain
from Orange.preprocess import Preprocess
from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors

from orangecontrib.spectroscopy.preprocess import SelectColumn, CommonDomain

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.widgets.gui import lineEditIntRange

from pySNOM.images import BackgroundPolyFit
import numpy as np


class AddFeature(SelectColumn):
    InheritEq = True


class _BackGroundFitCommon(CommonDomain):
    def __init__(self, xorder, yorder, domain):
        super().__init__(domain)
        self.xorder = xorder
        self.yorder = yorder

    def transformed(self, data):
        d, b = BackgroundPolyFit(xorder=self.xorder, yorder=self.yorder).transform(data.X)
        return d


class BackGroundFit(Preprocess):
    def __init__(self, xorder=1, yorder=1):
        self.xorder = xorder
        self.yorder = yorder

    def __call__(self, data):
        common = _BackGroundFitCommon(self.xorder, self.yorder, data.domain)
        atts = [
            a.copy(compute_value=AddFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class BackGroundFitEditor(BaseEditorOrange):
    name = "Polynomial background fit"
    qualname = "orangecontrib.snom.background_fit_test"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.xorder = 1
        self.yorder = 1

        form = QFormLayout()
        xorderedit = lineEditIntRange(self, self, "xorder", callback=self.edited.emit)
        yorderedit = lineEditIntRange(self, self, "yorder", callback=self.edited.emit)
        form.addRow("xorder", xorderedit)
        form.addRow("yorder", yorderedit)
        self.controlArea.setLayout(form)

    def activateOptions(self):
        pass  # actions when user starts changing options

    def setParameters(self, params):
        self.xorder = params.get("xorder", 1)
        self.yorder = params.get("yorder", 1)

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        xorder = float(params.get("xorder", 1))
        yorder = float(params.get("yorder", 1))
        return BackGroundFit(xorder=xorder, yorder=yorder)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(BackGroundFitEditor, 400)
