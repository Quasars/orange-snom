# this is just an example of registration

from AnyQt.QtWidgets import QFormLayout

from Orange.data import Domain
from Orange.preprocess import Preprocess
from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors

from orangecontrib.spectroscopy.preprocess import SelectColumn, CommonDomain

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatRange

from pySNOM.images import RotatePhase


class AddFeature(SelectColumn):
    InheritEq = True


class _PhaseRotCommon(CommonDomain):
    def __init__(self, degree, domain):
        super().__init__(domain)
        self.degree = degree

    def transformed(self, data):
        return RotatePhase(degree=self.degree).transform(data.X)


class PhaseRotation(Preprocess):
    def __init__(self, degree=0.0):
        self.degree = degree

    def __call__(self, data):
        common = _PhaseRotCommon(self.degree, data.domain)
        atts = [
            a.copy(compute_value=AddFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class PhaseRotationEditor(BaseEditorOrange):
    name = "Rotate phase"
    qualname = "orangecontrib.snom.phase_rotation_test"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.degree = 0.0

        form = QFormLayout()
        degreeedit = lineEditFloatRange(self, self, "degree", callback=self.edited.emit)
        # degreeedit = slideEditFloatRange(self, self, "degree", callback=self.edited.emit)
        form.addRow("degree", degreeedit)
        self.controlArea.setLayout(form)

    def activateOptions(self):
        pass  # actions when user starts changing options

    def setParameters(self, params):
        self.degree = params.get("degree", 0.0)

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        degree = float(params.get("degree", 0.0))
        return PhaseRotation(degree=degree)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(PhaseRotationEditor, 300)
