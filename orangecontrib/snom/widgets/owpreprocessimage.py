import time

from AnyQt.QtWidgets import QFormLayout
from orangewidget.settings import SettingProvider

import Orange.data
from Orange import preprocess
from Orange.preprocess import Preprocess
from Orange.widgets.data.owpreprocess import (
    PreprocessAction, Description, icon_path
)
from Orange.widgets.widget import Output

from orangecontrib.spectroscopy.preprocess import SelectColumn, \
    CommonDomain
from orangecontrib.spectroscopy.widgets.owhyper import ImagePlot

from orangecontrib.spectroscopy.widgets.owpreprocess import (
    GeneralPreprocess,
    create_preprocessor,
    InterruptException,
)
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

    def __init__(self, amount=0.):
        self.amount = amount

    def __call__(self, data):
        common = _AddCommon(self.amount, data.domain)
        atts = [a.copy(compute_value=AddFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class AddEditor(BaseEditorOrange):

    name = "Add constant"
    qualname = "orangecontrib.snom.add_constant_test"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.amount = 0.

        form = QFormLayout()
        amounte = lineEditFloatRange(self, self, "amount", callback=self.edited.emit)
        form.addRow("Addition", amounte)
        self.controlArea.setLayout(form)

    def activateOptions(self):
        pass  # actions when user starts changing options

    def setParameters(self, params):
        self.amount = params.get("amount", 0.)

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        amount = float(params.get("amount", 0.))
        return AddConstant(amount=amount)

    def set_preview_data(self, data):
        if data:
            pass  # TODO any settings


PREPROCESSORS = [
    PreprocessAction(
        c.name, c.qualname, "Image",
        Description(c.name, icon_path("Discretize.svg")),
        c
    ) for c in [
        AddEditor
    ]
]


class AImagePlot(ImagePlot):
    def clear_markings(self):
        pass


class ImagePreviews:
    curveplot = SettingProvider(AImagePlot)
    curveplot_after = SettingProvider(AImagePlot)

    value_type = 1

    def __init__(self):
        # the name of curveplot is kept because GeneralPreprocess
        # expects these names
        self.curveplot = AImagePlot(self)
        self.curveplot_after = AImagePlot(self)

    def shutdown(self):
        self.curveplot.shutdown()
        self.curveplot_after.shutdown()


class SpectralImagePreprocess(GeneralPreprocess, ImagePreviews, openclass=True):
    def __init__(self):
        ImagePreviews.__init__(self)
        super().__init__()

    def onDeleteWidget(self):
        super().onDeleteWidget()
        ImagePreviews.shutdown(self)


class OWPreprocessImage(SpectralImagePreprocess):
    name = "Preprocess image"
    id = "orangecontrib.snom.widgets.preprocessimage"
    description = "Process image"
    icon = "icons/preprocessimage.svg"
    priority = 1010

    settings_version = 2

    PREPROCESSORS = PREPROCESSORS
    BUTTON_ADD_LABEL = "Add integral..."

    class Outputs:
        preprocessed_data = Output("Integrated Data", Orange.data.Table, default=True)
        preprocessor = Output("Preprocessor", preprocess.preprocess.Preprocess)

    def __init__(self):
        self.markings_list = []
        super().__init__()

    def show_preview(self, show_info_anyway=False):
        # redraw integrals if number of preview curves was changed
        super().show_preview(False)

    def create_outputs(self):
        self._reference_compat_warning()
        pp_def = [
            self.preprocessormodel.item(i)
            for i in range(self.preprocessormodel.rowCount())
        ]
        self.start(
            self.run_task,
            self.data,
            self.reference_data,
            pp_def,
            self.process_reference,
        )

    @staticmethod
    def run_task(
        data: Orange.data.Table,
        reference: Orange.data.Table,
        pp_def,
        process_reference,
        state,
    ):
        def progress_interrupt(i: float):
            state.set_progress_value(i)
            if state.is_interruption_requested():
                raise InterruptException

        # Protects against running the task in succession many times, as would
        # happen when adding a preprocessor (there, commit() is called twice).
        # Wait 100 ms before processing - if a new task is started in meanwhile,
        # allow that is easily` cancelled.
        for _ in range(10):
            time.sleep(0.010)
            progress_interrupt(0)

        n = len(pp_def)
        plist = []
        for i in range(n):
            progress_interrupt(i / n * 100)
            item = pp_def[i]
            pp = create_preprocessor(item, reference)
            plist.append(pp)
            if data is not None:
                data = pp(data)
            progress_interrupt((i / n + 0.5 / n) * 100)
            if process_reference and reference is not None and i != n - 1:
                reference = pp(reference)
        # if there are no preprocessors, return None instead of an empty list
        preprocessor = preprocess.preprocess.PreprocessorList(plist) if plist else None
        return data, preprocessor


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWPreprocessImage).run(Orange.data.Table("iris.tab"))
