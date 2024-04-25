from orangecontrib.spectroscopy.preprocess import GaussianSmoothing

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors


class RegistrationExampleEditor(BaseEditorOrange):
    name = "Registration example"
    qualname = "orangecontrib.snom.registration_example"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

    def setParameters(self, params):
        pass

    @classmethod
    def createinstance(cls, params):
        return GaussianSmoothing(sd=10)


preprocess_editors.register(RegistrationExampleEditor, 1000)
