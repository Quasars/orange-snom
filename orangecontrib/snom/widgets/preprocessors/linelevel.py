from AnyQt.QtWidgets import QFormLayout

from Orange.data import Domain

from orangewidget.gui import comboBox

from pySNOM.images import LineLevel

from orangecontrib.spectroscopy.preprocess import SelectColumn, CommonDomain
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange

from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors
from orangecontrib.snom.widgets.preprocessors.utils import PreprocessImageOpts


class AddFeature(SelectColumn):
    InheritEq = True


class _LineLevelCommon(CommonDomain):
    def __init__(self, method, domain):
        super().__init__(domain)
        self.method = method

    def transformed(self, data):
        # TODO figure out 1D to 2D properly
        return LineLevel(method=self.method).transform(data.X)


class LineLevelProcessor(PreprocessImageOpts):
    def __init__(self, method="median"):
        self.method = method

    def __call__(self, data, image_opts):
        common = _LineLevelCommon(self.method, data.domain)
        atts = [
            a.copy(compute_value=AddFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


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
