from AnyQt.QtWidgets import QFormLayout

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.widgets.gui import lineEditIntRange

from pySNOM.images import BackgroundPolyFit, DataTypes

from orangecontrib.snom.preprocess.utils import (
    PreprocessImageOpts2DOnlyWhole,
)
from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors


class BackGroundFit(PreprocessImageOpts2DOnlyWhole):
    def __init__(self, xorder=1, yorder=1):
        self.xorder = xorder
        self.yorder = yorder

    def transform_image(self, image, data):
        datatype = data.attributes.get("measurement.signaltype", "Phase")
        d, b = BackgroundPolyFit(
            xorder=self.xorder, yorder=self.yorder, datatype=DataTypes[datatype]
        ).transform(image)
        return d


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
