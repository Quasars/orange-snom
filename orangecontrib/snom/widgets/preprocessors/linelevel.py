import numpy as np
from AnyQt.QtWidgets import QFormLayout

from orangewidget.gui import comboBox, checkBox

from pySNOM.images import LineLevel, DataTypes

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange

from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors
from orangecontrib.snom.preprocess.utils import (
    PreprocessImageOpts2DOnlyWhole,
)


class LineLevelProcessor(PreprocessImageOpts2DOnlyWhole):
    def __init__(self, method="median", use_mask=False):
        self.method = method
        self.use_mask = use_mask

    def transform_image(self, image, data, mask=None):
        datatype = data.attributes.get("measurement.signaltype", "Phase")
        mask = mask if self.use_mask else None
        processed = LineLevel(
            method=self.method, datatype=DataTypes[datatype]
        ).transform(image,mask=mask)
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
        self.use_mask = False

        form = QFormLayout()
        self.levelmethod_cb = comboBox(self, self, "method", callback=self.setmethod)
        self.levelmethod_cb.addItems(['median', 'mean', 'difference'])
        form.addRow("Leveling method", self.levelmethod_cb)

        self.use_mask_chb = checkBox(self, self,"use_mask","Enable",callback=self.edited.emit)
        form.addRow("Masking", self.use_mask_chb)
        self.controlArea.setLayout(form)

    def activateOptions(self):
        pass  # actions when user starts changing options

    def setmethod(self):
        self.method = self.levelmethod_cb.currentText()
        self.edited.emit()

    def setParameters(self, params):
        self.method = params.get("method", "median")
        self.use_mask = params.get("use_mask", False)

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        method = params.get("method", "median")
        use_mask = bool(params.get("use_mask", False))
        return LineLevelProcessor(method=method,use_mask=use_mask)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(LineLevelEditor, 200)
