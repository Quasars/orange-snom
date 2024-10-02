from AnyQt.QtWidgets import QFormLayout

from orangewidget.gui import comboBox

from pySNOM.images import LineLevel

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange

from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors
from orangecontrib.snom.preprocess.utils import (
    PreprocessImageOpts2DOnlyWhole,
)


class LineLevelProcessor(PreprocessImageOpts2DOnlyWhole):
    def __init__(self, method="median"):
        self.method = method

    def transform_image(self, image, data):
        return LineLevel(method=self.method).transform(image)


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
