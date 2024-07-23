from AnyQt.QtWidgets import QVBoxLayout

from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.snom.preprocess.phase_unwrap import PhaseUnwrap


class PhaseUnwrapEditor(BaseEditorOrange):
    """
    Phase Unwrap editor.
    """

    name = "Phase Unwrap"
    qualname = "orangecontrib.snom.phaseunwrap"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.controlArea.setLayout(QVBoxLayout())

    def setParameters(self, params):
        pass

    @classmethod
    def createinstance(cls, params):
        return PhaseUnwrap()


preprocess_editors.register(PhaseUnwrapEditor, 1000)
