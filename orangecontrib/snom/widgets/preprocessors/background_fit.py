from AnyQt.QtWidgets import QFormLayout

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.widgets.gui import lineEditIntRange
from orangewidget.gui import checkBox

from pySNOM.images import MaskedBackgroundPolyFit, DataTypes

from orangecontrib.snom.preprocess.utils import (
    PreprocessImageOpts2DOnlyWhole,
)
from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors


class BackGroundFit(PreprocessImageOpts2DOnlyWhole):
    def __init__(self, xorder=1, yorder=1, use_mask=False):
        self.xorder = xorder
        self.yorder = yorder
        self.use_mask = use_mask

    def transform_image(self, image, data, mask=None):
        datatype = data.attributes.get("measurement.signaltype", "Phase")
        mask = mask if self.use_mask else None
        d = MaskedBackgroundPolyFit(
            xorder=self.xorder, yorder=self.yorder, datatype=DataTypes[datatype]
        ).transform(image,mask=mask)
        return d


class BackGroundFitEditor(BaseEditorOrange):
    name = "Polynomial background fit"
    qualname = "orangecontrib.snom.background_fit_test"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.xorder = 1
        self.yorder = 1
        self.use_mask = False

        form = QFormLayout()
        xorderedit = lineEditIntRange(self, self, "xorder", callback=self.edited.emit)
        yorderedit = lineEditIntRange(self, self, "yorder", callback=self.edited.emit)
        self.use_mask_chb = checkBox(self, self,"use_mask","Enable",callback=self.edited.emit)
        
        form.addRow("xorder", xorderedit)
        form.addRow("yorder", yorderedit)
        form.addRow("Masking", self.use_mask_chb)

        self.controlArea.setLayout(form)

    def activateOptions(self):
        pass  # actions when user starts changing options

    def setParameters(self, params):
        self.xorder = params.get("xorder", 1)
        self.yorder = params.get("yorder", 1)
        self.use_mask = params.get("use_mask", False)

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        xorder = float(params.get("xorder", 1))
        yorder = float(params.get("yorder", 1))
        use_mask = bool(params.get("use_mask", False))
        return BackGroundFit(xorder=xorder, yorder=yorder, use_mask=use_mask)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(BackGroundFitEditor, 400)
