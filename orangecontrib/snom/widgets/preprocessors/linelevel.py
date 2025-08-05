import numpy as np
from AnyQt.QtWidgets import QFormLayout

from orangewidget.gui import comboBox, checkBox

from pySNOM.images import LineLevel, DataTypes

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange

from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors
from orangecontrib.snom.preprocess.utils import (
    PreprocessImageOpts2DOnlyWhole,
    MaskOptions,
    transform_mask
)


class LineLevelProcessor(PreprocessImageOpts2DOnlyWhole):
    def __init__(self, method="median", mask_method=False):
        self.method = method
        self.mask_method = mask_method

    def transform_image(self, image, data, mask=None):
        datatype = data.attributes.get("measurement.signaltype", "Phase")
        mask = transform_mask(mask=mask, option=MaskOptions[self.mask_method])
        processed = LineLevel(
            method=self.method, datatype=DataTypes[datatype]
        ).transform(image, mask=mask)
        if self.method == 'difference':
            # add a row of NaN, so that the size matches
            processed = np.vstack((processed, np.full((1, image.shape[1]), np.nan)))
        return processed


class LineLevelEditor(BaseEditorOrange):
    name = "Line leveling"
    qualname = "orangecontrib.snom.line_level_test"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.method = 'median'
        self.mask_method = 'IGNORE'
        self.use_mask = False

        form = QFormLayout()
        self.levelmethod_cb = comboBox(self, self, "method", callback=self.setmethod)
        self.levelmethod_cb.addItems(['median', 'mean', 'difference'])
        form.addRow("Leveling method", self.levelmethod_cb)

        self.maskmethod_cb = comboBox(self,self,"mask_method",callback=self.setmethod)
        self.maskmethod_cb.addItems([e.name for e in MaskOptions])
        form.addRow("Mask", self.maskmethod_cb)
        self.controlArea.setLayout(form)

    def activateOptions(self):
        pass  # actions when user starts changing options

    def setmethod(self):
        self.method = self.levelmethod_cb.currentText()
        self.mask_method = self.maskmethod_cb.currentText()
        self.edited.emit()

    def setParameters(self, params):
        self.method = params.get("method", "median")
        self.mask_method = params.get("mask_method", "IGNORE")

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        method = params.get("method", "median")
        mask_method = params.get("mask_method", "IGNORE")
        return LineLevelProcessor(method=method, mask_method=mask_method)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(LineLevelEditor, 200)
