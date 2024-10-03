from AnyQt.QtWidgets import QFormLayout, QLabel

from orangecontrib.spectroscopy.preprocess import MissingReferenceException
from orangecontrib.spectroscopy.widgets.preprocessors.utils import (
    BaseEditorOrange,
    REFERENCE_DATA_PARAM,
)

from pySNOM.images import SelfReference, DataTypes

from orangecontrib.snom.preprocess.utils import (
    PreprocessImageOpts2DOnlyWholeReference,
)
from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors


class SelfRef(PreprocessImageOpts2DOnlyWholeReference):
    def __init__(self, reference):
        self.reference = reference
        if self.reference is None:
            raise MissingReferenceException("Self-referencing needs a reference")

    def transform_image(self, image, ref_image, data):
        datatype = data.attributes.get("measurement.signaltype", "Phase")
        return SelfReference(
            referencedata=ref_image, datatype=DataTypes[datatype]
        ).transform(image)


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
        self.update_reference_info()

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

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        reference = params.get(REFERENCE_DATA_PARAM, None)
        return SelfRef(reference=reference)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


preprocess_image_editors.register(SelfRefEditor, 400)
