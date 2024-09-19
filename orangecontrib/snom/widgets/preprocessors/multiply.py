# this is just an example of registration

from AnyQt.QtWidgets import QFormLayout

from Orange.data import Domain
from Orange.preprocess import Preprocess
from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors

from orangecontrib.spectroscopy.preprocess import SelectColumn, CommonDomain

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatRange


class AddFeature(SelectColumn):
    InheritEq = True


class _AddCommon(CommonDomain):
    def __init__(self, amount, domain):
        super().__init__(domain)
        self.amount = amount

    def transformed(self, data):
        return data.X + self.amount


class AddConstant(Preprocess):
    def __init__(self, amount=0.0):
        self.amount = amount

    def __call__(self, data):
        common = _AddCommon(self.amount, data.domain)
        atts = [
            a.copy(compute_value=AddFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class AddEditor(BaseEditorOrange):
    name = "Add constant"
    qualname = "orangecontrib.snom.add_constant_test"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.amount = 0.0

        form = QFormLayout()
        amounte = lineEditFloatRange(self, self, "amount", callback=self.edited.emit)
        form.addRow("Addition", amounte)
        self.controlArea.setLayout(form)

    def activateOptions(self):
        pass  # actions when user starts changing options

    def setParameters(self, params):
        self.amount = params.get("amount", 0.0)

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        amount = float(params.get("amount", 0.0))
        return AddConstant(amount=amount)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(AddEditor, 100)
