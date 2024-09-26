from AnyQt.QtWidgets import QFormLayout

from Orange.data import Domain
from Orange.preprocess import Preprocess
from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors

from orangecontrib.spectroscopy.preprocess import SelectColumn, CommonDomain

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatRange
from orangewidget.gui import comboBox

from pySNOM.images import SimpleNormalize


class AddFeature(SelectColumn):
    InheritEq = True


class _SimpleNormCommon(CommonDomain):
    def __init__(self, method, value, domain):
        super().__init__(domain)
        self.method = method
        self.value = value
        # print(value,method)

    def transformed(self, data):
        return SimpleNormalize(method=self.method, value=self.value).transform(data.X)


class SimpleNorm(Preprocess):
    def __init__(self, method='median', value=1.0):
        self.method = method
        self.value = value

    def __call__(self, data):
        common = _SimpleNormCommon(self.method, self.value, data.domain)
        atts = [
            a.copy(compute_value=AddFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class SimpleNormEditor(BaseEditorOrange):
    name = "Simple normalization"
    qualname = "orangecontrib.snom.simple_normalize"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.method = "manual"
        self.value = 1.0

        form = QFormLayout()
        self.valueedit = lineEditFloatRange(
            self, self, "value", callback=self.edited.emit
        )
        self.cb_method = comboBox(self, self, "method", callback=self.setmethod)
        self.cb_method.addItems(['median', 'mean', 'manual'])
        self.cb_method.setCurrentText('manual')
        form.addRow("method", self.cb_method)
        form.addRow("value", self.valueedit)
        self.controlArea.setLayout(form)

    def setmethod(self):
        if self.cb_method.currentText() != "manual":
            self.valueedit.setEnabled(False)
        else:
            self.valueedit.setEnabled(True)

        self.method = self.cb_method.currentText()
        self.edited.emit()

    def activateOptions(self):
        pass  # actions when user starts changing options

    def setParameters(self, params):
        self.method = params.get("method", "manual")
        self.value = params.get("value", 1)

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        method = str(params.get("method", "manual"))
        value = float(params.get("value", 1))
        return SimpleNorm(method=method, value=value)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(SimpleNormEditor, 500)
