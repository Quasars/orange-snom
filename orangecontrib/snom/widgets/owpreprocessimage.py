import time

from AnyQt.QtCore import Qt

from Orange.data import Domain, DiscreteVariable, ContinuousVariable
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel
from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors
from orangecontrib.snom.preprocess.utils import PreprocessImageOpts
from orangewidget import gui
from orangewidget.settings import SettingProvider, ContextSetting, Setting

import Orange.data
from Orange import preprocess
from Orange.widgets.widget import Output, Input

from orangecontrib.spectroscopy.widgets.owhyper import BasicImagePlot

from orangecontrib.spectroscopy.widgets.owpreprocess import (
    GeneralPreprocess,
    create_preprocessor,
    InterruptException,
    PreviewRunner,
)

from orangewidget.widget import Msg


class AImagePlot(BasicImagePlot):
    attr_x = None  # not settings, set from the parent class
    attr_y = None

    def __init__(self, parent):
        super().__init__(parent)
        self.axes_settings_box.hide()
        self.rgb_settings_box.hide()

    def add_selection_actions(self, _):
        pass

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


def execute_with_image_opts(pp, data, image_opts):
    if isinstance(pp, PreprocessImageOpts):
        return pp(data, image_opts)
    return pp(data)


class ImagePreviewRunner(PreviewRunner):
    def show_preview(self, show_info_anyway=False):
        """Shows preview and also passes preview data to the widgets"""
        master = self.master
        self.preview_pos = master.flow_view.preview_n()
        self.last_partial = None
        self.show_info_anyway = show_info_anyway
        self.preview_data = None
        self.after_data = None
        pp_def = [
            master.preprocessormodel.item(i)
            for i in range(master.preprocessormodel.rowCount())
        ]
        if master.data is not None:
            data = master.sample_data(master.data)
            image_opts = master.image_opts()
            self.start(
                self.run_preview,
                data,
                master.reference_data,
                image_opts,
                pp_def,
                master.process_reference,
            )
        else:
            master.curveplot.set_data(None)
            master.curveplot_after.set_data(None)

    @staticmethod
    def run_preview(
        data: Orange.data.Table,
        reference: Orange.data.Table,
        image_opts,
        pp_def,
        process_reference,
        state,
    ):
        def progress_interrupt(i: float):
            if state.is_interruption_requested():
                raise InterruptException

        n = len(pp_def)
        orig_data = data
        for i in range(n):
            progress_interrupt(0)
            state.set_partial_result((i, data, reference))
            item = pp_def[i]
            pp = create_preprocessor(item, reference)
            data = execute_with_image_opts(pp, data, image_opts)
            progress_interrupt(0)
            if process_reference and reference is not None and i != n - 1:
                reference = execute_with_image_opts(pp, reference, image_opts)
        progress_interrupt(0)
        state.set_partial_result((n, data, None))
        return orig_data, data


class SpectralImagePreprocessReference(SpectralImagePreprocess, openclass=True):
    class Inputs(SpectralImagePreprocess.Inputs):
        reference = Input("Reference", Orange.data.Table)

    @Inputs.reference
    def set_reference(self, reference):
        self.reference_data = reference


class OWPreprocessImage(SpectralImagePreprocessReference):
    name = "Preprocess image"
    id = "orangecontrib.snom.widgets.preprocessimage"
    description = "Process image"
    icon = "icons/preprocessimage.svg"
    priority = 1010

    settings_version = 2

    settingsHandler = DomainContextHandler()

    _max_preview_spectra = 1000000
    preview_curves = Setting(100000)

    editor_registry = preprocess_image_editors
    BUTTON_ADD_LABEL = "Add preprocessor..."

    attr_value = ContextSetting(None)
    attr_x = ContextSetting(None, exclude_attributes=True)
    attr_y = ContextSetting(None, exclude_attributes=True)

    class Outputs:
        preprocessed_data = Output("Integrated Data", Orange.data.Table, default=True)
        preprocessor = Output("Preprocessor", preprocess.preprocess.Preprocess)

    class Warning(SpectralImagePreprocess.Warning):
        threshold_error = Msg("Low slider should be less than High")

    class Error(SpectralImagePreprocess.Error):
        image_too_big = Msg("Image for chosen features is too big ({} x {}).")

    class Information(SpectralImagePreprocess.Information):
        not_shown = Msg("Undefined positions: {} data point(s) are not shown.")

    def image_values(self):
        attr_value = self.attr_value.name if self.attr_value else None
        return lambda data, attr=attr_value: data.transform(Domain([data.domain[attr]]))

    def image_values_fixed_levels(self):
        return None

    def __init__(self):
        self.markings_list = []
        super().__init__()

        self.preview_runner = ImagePreviewRunner(self)

        self.feature_value_model = DomainModel(
            order=(
                DomainModel.ATTRIBUTES,
                DomainModel.Separator,
                DomainModel.CLASSES,
                DomainModel.Separator,
                DomainModel.METAS,
            ),
            valid_types=ContinuousVariable,
        )

        common_options = {
            "labelWidth": 50,
            "orientation": Qt.Horizontal,
            "sendSelectedValue": True,
        }

        self.feature_value = gui.comboBox(
            self.preview_settings_box,
            self,
            "attr_value",
            label="Show",
            contentsLength=12,
            searchable=True,
            callback=self.update_feature_value,
            model=self.feature_value_model,
            **common_options
        )

        self.xy_model = DomainModel(
            DomainModel.METAS | DomainModel.CLASSES, valid_types=DomainModel.PRIMITIVE
        )

        self.cb_attr_x = gui.comboBox(
            self.preview_settings_box,
            self,
            "attr_x",
            label="Axis x",
            callback=self.update_attr,
            model=self.xy_model,
            **common_options
        )
        self.cb_attr_y = gui.comboBox(
            self.preview_settings_box,
            self,
            "attr_y",
            label="Axis y",
            callback=self.update_attr,
            model=self.xy_model,
            **common_options
        )

        self.contextAboutToBeOpened.connect(lambda x: self.init_interface_data(x[0]))

        self.preview_runner.preview_updated.connect(self.redraw_data)

    def update_attr(self):
        self.curveplot.attr_x = self.attr_x
        self.curveplot.attr_y = self.attr_y
        self.curveplot_after.attr_x = self.attr_x
        self.curveplot_after.attr_y = self.attr_y
        self.on_modelchanged()

    def update_feature_value(self):
        self.on_modelchanged()

    def redraw_data(self):
        self.curveplot.update_view()
        self.curveplot_after.update_view()

    def init_interface_data(self, data):
        self.init_attr_values(data)
        self.curveplot.init_interface_data(data)
        self.curveplot_after.init_interface_data(data)

    def init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.feature_value_model.set_domain(domain)
        self.attr_value = (
            self.feature_value_model[0] if self.feature_value_model else None
        )
        self.xy_model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 else self.attr_x

    def image_opts(self):
        return {
            'attr_x': str(self.attr_x),
            'attr_y': str(self.attr_y),
            'attr_value': str(self.attr_value),
        }

    def create_outputs(self):
        self._reference_compat_warning()
        pp_def = [
            self.preprocessormodel.item(i)
            for i in range(self.preprocessormodel.rowCount())
        ]
        image_opts = self.image_opts()
        self.start(
            self.run_task,
            self.data,
            self.reference_data,
            image_opts,
            pp_def,
            self.process_reference,
        )

    def set_data(self, data):
        self.curveplot.set_data(None)
        self.curveplot_after.set_data(None)

        super().set_data(data)

        self.closeContext()

        def valid_context(data):
            if data is None:
                return False
            annotation_features = [
                v
                for v in data.domain.metas + data.domain.class_vars
                if isinstance(v, (DiscreteVariable, ContinuousVariable))
            ]
            return len(annotation_features) >= 1

        if valid_context(data):
            self.openContext(data)
        else:
            # to generate valid interface even if context was not loaded
            self.contextAboutToBeOpened.emit([data])

        self.update_attr()  # update imageplots attributes from the master

    @staticmethod
    def run_task(
        data: Orange.data.Table,
        reference: Orange.data.Table,
        image_opts,
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
                data = execute_with_image_opts(pp, data, image_opts)
            progress_interrupt((i / n + 0.5 / n) * 100)
            if process_reference and reference is not None and i != n - 1:
                reference = execute_with_image_opts(pp, reference, image_opts)
        # if there are no preprocessors, return None instead of an empty list
        preprocessor = preprocess.preprocess.PreprocessorList(plist) if plist else None
        return data, preprocessor


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWPreprocessImage).run(Orange.data.Table("whitelight.gsf"))
