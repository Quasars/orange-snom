import numpy as np
from AnyQt.QtWidgets import QFormLayout

from orangewidget.gui import comboBox

from pySNOM.images import LineLevel, DataTypes

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange

from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors
from orangecontrib.snom.preprocess.utils import (
    PreprocessImageOpts2DOnlyWhole,
)


class LineLevelProcessor(PreprocessImageOpts2DOnlyWhole):
    def __init__(self, method="median"):
        self.method = method

    def transform_image(self, image, data):
        datatype = data.attributes.get("measurement.signaltype", "Phase")
        processed = LineLevel(
            method=self.method, datatype=DataTypes[datatype]
        ).transform(image)
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

        form = QFormLayout()
        self.levelmethod_cb = comboBox(self, self, "method", callback=self.setmethod)
        self.levelmethod_cb.addItems(['median', 'mean', 'difference'])
        form.addRow("Leveling method", self.levelmethod_cb)
        self.controlArea.setLayout(form)

    def activateOptions(self):
        pass  # actions when user starts changing options

    def setmethod(self):
        self.method = self.levelmethod_cb.currentText()
        self.edited.emit()

    def setParameters(self, params):
        self.method = params.get("method", "median")

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        method = params.get("method", "median")
        return LineLevelProcessor(method=method)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(LineLevelEditor, 200)
