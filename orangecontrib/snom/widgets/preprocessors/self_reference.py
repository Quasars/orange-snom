# this is just an example of registration

from AnyQt.QtWidgets import QFormLayout, QLabel

from Orange.data import Domain
from Orange.preprocess import Preprocess
from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors

from orangecontrib.spectroscopy.preprocess import SelectColumn, CommonDomain

from orangecontrib.spectroscopy.widgets.preprocessors.utils import (
    BaseEditorOrange,
    REFERENCE_DATA_PARAM,
)

from pySNOM.images import SelfReference


class AddFeature(SelectColumn):
    InheritEq = True


class _SelfRefCommon(CommonDomain):
    def __init__(self, reference, domain):
        super().__init__(domain)
        self.reference = reference
        # print(value,method)

    def transformed(self, data):
        if self.reference:
            return SelfReference(referencedata=2 * self.reference.X).transform(data.X)
        else:
            return data.X


class SelfRef(Preprocess):
    def __init__(self, reference):
        self.reference = reference

    def __call__(self, data):
        common = _SelfRefCommon(self.reference, data.domain)
        atts = [
            a.copy(compute_value=AddFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class SelfRefEditor(BaseEditorOrange):
    name = "Self-referencing"
    qualname = "orangecontrib.snom.self_reference"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.reference = None

        form = QFormLayout()
        self.reference_info = QLabel("Reference data from input!")
        form.addRow(self.reference_info)
        self.controlArea.setLayout(form)

    def activateOptions(self):
        pass  # actions when user starts changing options

    def setParameters(self, params):
        # self.refdata = params.get("refdata", None)
        self.update_reference_info()

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        reference = params.get(REFERENCE_DATA_PARAM, None)
        return SelfRef(reference=reference)

    def set_reference_data(self, reference):
        self.reference = reference
        self.update_reference_info()

    def update_reference_info(self):
        if not self.reference:
            self.reference_info.setText("Reference: missing!")
            self.reference_info.setStyleSheet("color: red")
        else:
            rinfo = "Reference order: N"
            self.reference_info.setText(rinfo)
            self.reference_info.setStyleSheet("color: black")

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(SelfRefEditor, 600)
