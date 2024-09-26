from AnyQt.QtWidgets import QFormLayout

from Orange.data import Domain
from orangecontrib.spectroscopy.utils import get_hypercube

from orangewidget.gui import comboBox

from pySNOM.images import LineLevel

from orangecontrib.spectroscopy.preprocess import SelectColumn, CommonDomain
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange

from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors
from orangecontrib.snom.widgets.preprocessors.utils import PreprocessImageOpts


class _LineLevelCommon(CommonDomain):
    def __init__(self, method, domain, image_opts):
        super().__init__(domain)
        self.method = method
        self.image_opts = image_opts

    def transformed(self, data):
        vat = data.domain[self.image_opts["attr_value"]]
        ndom = Domain([vat], data.domain.class_vars, data.domain.metas)
        data = data.transform(ndom)
        xat = data.domain[self.image_opts["attr_x"]]
        yat = data.domain[self.image_opts["attr_y"]]
        hypercube, lsx, lsy = get_hypercube(data, xat, yat)
        transformed = LineLevel(method=self.method).transform(hypercube[:, :, 0])
        print(transformed)
        # TODO transform the resulting matrix back to original indices;
        # for this the get_hypercube will need to return an actual index matrix


class LineLevelProcessor(PreprocessImageOpts):
    def __init__(self, method="median"):
        self.method = method

    def __call__(self, data, image_opts):
        common = _LineLevelCommon(self.method, data.domain, image_opts)
        at = data.domain[image_opts["attr_value"]].copy(
            compute_value=SelectColumn(0, common)
        )
        domain = Domain([at], data.domain.class_vars, data.domain.metas)
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
