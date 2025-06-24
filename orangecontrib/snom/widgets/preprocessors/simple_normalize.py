from AnyQt.QtWidgets import QFormLayout

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatRange
from orangewidget.gui import comboBox, checkBox

from pySNOM.images import SimpleNormalize, DataTypes

from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors
from orangecontrib.snom.preprocess.utils import (
    PreprocessImageOpts2DOnlyWhole,
)


class SimpleNorm(PreprocessImageOpts2DOnlyWhole):
    def __init__(self, method, value, use_mask=False):
        self.method = method
        self.value = value
        self.use_mask = use_mask

    def transform_image(self, image, data, mask=None):
        datatype = data.attributes.get("measurement.signaltype", "Phase")
        mask = mask if self.use_mask else None
        return SimpleNormalize(
            method=self.method, value=self.value, datatype=DataTypes[datatype]
        ).transform(image, mask=mask)


class SimpleNormEditor(BaseEditorOrange):
    name = "Simple normalization"
    qualname = "orangecontrib.snom.simple_normalize"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.method = "manual"
        self.value = 1.0
        self.use_mask = False

        form = QFormLayout()
        self.valueedit = lineEditFloatRange(
            self, self, "value", callback=self.edited.emit
        )
        self.cb_method = comboBox(self, self, "method", callback=self.setmethod)
        self.cb_method.addItems(['median', 'mean', 'manual'])
        self.cb_method.setCurrentText('manual')
        self.use_mask_chb = checkBox(self, self,"use_mask","Enable",callback=self.edited.emit)

        form.addRow("Method", self.cb_method)
        form.addRow("Value", self.valueedit)
        form.addRow("Masking", self.use_mask_chb)
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
        self.use_mask = params.get("use_mask", False)

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        method = str(params.get("method", "manual"))
        value = float(params.get("value", 1))
        use_mask = bool(params.get("use_mask", False))
        return SimpleNorm(method=method, value=value, use_mask=use_mask)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(SimpleNormEditor, 500)
