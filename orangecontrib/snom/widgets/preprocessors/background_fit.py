from AnyQt.QtWidgets import QFormLayout

from orangewidget.gui import comboBox

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.widgets.gui import lineEditIntRange
from orangewidget.gui import checkBox

from pySNOM.images import MaskedBackgroundPolyFit, DataTypes

from orangecontrib.snom.preprocess.utils import (
    PreprocessImageOpts2DOnlyWhole,
    MaskOptions,
    transform_mask,
)
from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors


class BackGroundFit(PreprocessImageOpts2DOnlyWhole):
    def __init__(self, xorder=1, yorder=1, mask_method=False):
        self.xorder = xorder
        self.yorder = yorder
        self.mask_method = mask_method

    def transform_image(self, image, data, mask=None):
        datatype = data.attributes.get("measurement.signaltype", "Phase")
        mask = transform_mask(mask=mask, option=MaskOptions[self.mask_method])
        d = MaskedBackgroundPolyFit(
            xorder=self.xorder, yorder=self.yorder, datatype=DataTypes[datatype]
        ).transform(image, mask=mask)
        return d


class BackGroundFitEditor(BaseEditorOrange):
    name = "Polynomial background fit"
    qualname = "orangecontrib.snom.background_fit_test"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.xorder = 1
        self.yorder = 1
        self.mask_method = False

        form = QFormLayout()
        xorderedit = lineEditIntRange(self, self, "xorder", callback=self.edited.emit)
        yorderedit = lineEditIntRange(self, self, "yorder", callback=self.edited.emit)
        self.maskmethod_cb = comboBox(
            self, self, "mask_method", callback=self.setmethod
        )
        self.maskmethod_cb.addItems([e.name for e in MaskOptions])

        form.addRow("xorder", xorderedit)
        form.addRow("yorder", yorderedit)
        form.addRow("Mask", self.maskmethod_cb)

        self.controlArea.setLayout(form)

    def activateOptions(self):
        pass  # actions when user starts changing options
    
    def setmethod(self):
        self.mask_method = self.maskmethod_cb.currentText()
        self.edited.emit()

    def setParameters(self, params):
        self.xorder = params.get("xorder", 1)
        self.yorder = params.get("yorder", 1)
        self.mask_method = params.get("mask_method", "IGNORE")

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        xorder = float(params.get("xorder", 1))
        yorder = float(params.get("yorder", 1))
        mask_method = params.get("mask_method", "IGNORE")
        return BackGroundFit(xorder=xorder, yorder=yorder, mask_method=mask_method)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(BackGroundFitEditor, 400)
