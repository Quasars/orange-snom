import multiprocessing
import concurrent.futures
import sys
import time

import numpy as np
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.data.owpreprocess import PreprocessAction, Description, icon_path
from Orange.widgets.data.utils.preprocess import DescriptionRole
from Orange.widgets.utils.concurrent import TaskState
from lmfit import Model, Parameters
from lmfit.model import ModelResult
from orangecontrib.spectroscopy.widgets.owpreprocess import InterruptException
from orangecontrib.spectroscopy.widgets.peakfit_compute import (
    pool_fit2,
)
from orangewidget.utils.widgetpreview import WidgetPreview

from orangecontrib.spectroscopy.preprocess import Cut
from orangecontrib.spectroscopy.preprocess.integrate import (
    INTEGRATE_DRAW_BASELINE_PENARGS,
    INTEGRATE_DRAW_CURVE_PENARGS,
)
from orangecontrib.spectroscopy.util import getx
from orangecontrib.spectroscopy.widgets.owhyper import refresh_integral_markings
from orangecontrib.spectroscopy.widgets.owpeakfit import (
    OWPeakFit,
    create_model,
    prepare_params,
    PeakPreviewRunner,
    unique_prefix,
)
from orangecontrib.spectroscopy.widgets.peak_editors import (
    ModelEditor,
    ConstantModelEditor,
)

from orangecontrib.snom.model.snompy import (
    compose_model,
    LorentzianPermittivityModel,
    DrudePermittivityModel,
)


class StaticPermittivityEditor(ConstantModelEditor):
    name = "Static Permittivity"


class InterfaceEditor(ConstantModelEditor):
    name = "Interface"
    prefix_generic = 'if'


class LorentzianPermittivityEditor(ModelEditor):
    name = "Lorentzian Permittivity"
    model = LorentzianPermittivityModel
    prefix_generic = 'lp'
    category = "Peak"
    icon = "Normalize.svg"

    @staticmethod
    def model_parameters():
        return 'nu_j', 'A_j', 'gamma_j', 'eps_inf'

    @staticmethod
    def model_lines():
        return ('nu_j',)


class DrudePermittivityEditor(ModelEditor):
    name = "Drude Permittivity"
    model = DrudePermittivityModel
    prefix_generic = 'dp'
    category = "Peak"
    icon = "Normalize.svg"

    @staticmethod
    def model_parameters():
        return 'nu_plasma', 'gamma', 'eps_inf'

    # Plotting the plasma wavenumber doesn't make sense
    # @staticmethod
    # def model_lines():
    #     return ('nu_plasma',)


def pack_model_editor(editor):
    return PreprocessAction(
        name=editor.name,
        qualname=f"orangecontrib.spectroscopy.widgets.owsnompy.{editor.prefix_generic}",
        category=editor.category,
        description=Description(
            getattr(editor, 'description', editor.name), icon_path(editor.icon)
        ),
        viewclass=editor,
    )


PREPROCESSORS = [
    pack_model_editor(e)
    for e in [
        LorentzianPermittivityEditor,
        StaticPermittivityEditor,
    ]
]


class ComplexPeakPreviewRunner(PeakPreviewRunner):
    def on_done(self, result):
        orig_data, after_data, model_result = result
        final_preview = self.preview_pos is None
        if final_preview:
            self.preview_data = orig_data
            self.after_data = orig_data

        if self.preview_data is None:  # happens in OWIntegrate
            self.preview_data = orig_data

        self.preview_model_result = model_result

        # TODO handle complex input data here?
        self.master.curveplot.set_data(self.preview_data)
        self.master.curveplot_after.set_data(self.preview_data)

        self.show_image_info(final_preview)

        self.preview_updated.emit()

    # Identical to parent except
    # -- calls model list
    # -- different compute function
    @staticmethod
    def run_preview(data: Table, m_def, pool, state: TaskState):
        def progress_interrupt(_: float):
            if state.is_interruption_requested():
                raise InterruptException

        # Protects against running the task in succession many times, as would
        # happen when adding a preprocessor (there, commit() is called twice).
        # Wait 100 ms before processing - if a new task is started in meanwhile,
        # allow that is easily` cancelled.
        for _ in range(10):
            time.sleep(0.010)
            progress_interrupt(0)

        orig_data = data

        model_list, parameters = create_model_list(m_def)
        model = compose_model(model_list)

        model_result = {}
        x = getx(data)
        if data is not None and model is not None:
            for row in data:
                progress_interrupt(0)
                res = pool.schedule(pool_fit2, (row.x, model.dumps(), parameters, x))
                while not res.done():
                    try:
                        progress_interrupt(0)
                    except InterruptException:
                        # CANCEL
                        if (
                            multiprocessing.get_start_method() != "fork"
                            and res.running()
                        ):
                            # If slower start methods are used, give the current computation
                            # some time to exit gracefully; this avoids reloading processes
                            concurrent.futures.wait([res], 1.0)
                        if not res.done():
                            res.cancel()
                        raise
                    concurrent.futures.wait([res], 0.05)
                fits = res.result()
                model_result[row.id] = ModelResult(model, parameters).loads(fits)

        progress_interrupt(0)
        return orig_data, data, model_result


class OWSnomModel(OWPeakFit):
    name = "SNOM Model"
    description = "Model SNOM spectra with snompy"

    PREPROCESSORS = PREPROCESSORS
    BUTTON_ADD_LABEL = "Add term..."

    def __init__(self):
        super().__init__()

        # Show _after (work-around for complex plotting)
        self.curveplot_after.show()
        # Markings list for _after
        self.markings_list_after = []

        # Custom preview running just to plot complex values
        self.preview_runner = ComplexPeakPreviewRunner(self)

        # Model options
        gui.widgetBox(self.controlArea, "Model Options")

    def redraw_integral(self):
        dis_abs = []
        dis_angle = []
        if self.curveplot.data:
            x = np.sort(getx(self.curveplot.data))
            previews = self.flow_view.preview_n()
            for i in range(self.preprocessormodel.rowCount()):
                if i in previews:
                    item = self.preprocessormodel.item(i)
                    m = create_model(item, i)
                    p = prepare_params(item, m)
                    # Show initial fit values for now
                    init = np.atleast_2d(np.broadcast_to(m.eval(p, x=x), x.shape))
                    init_abs = np.abs(init)
                    init_angle = np.angle(init)
                    di_abs = [("curve", (x, init_abs, INTEGRATE_DRAW_BASELINE_PENARGS))]
                    di_angle = [
                        ("curve", (x, init_angle, INTEGRATE_DRAW_BASELINE_PENARGS))
                    ]
                    color = self.flow_view.preview_color(i)
                    dis_abs.append({"draw": di_abs, "color": color})
                    dis_angle.append({"draw": di_angle, "color": color})
        result = None
        if (
            np.any(self.curveplot.selection_group)
            and self.curveplot.data
            and self.preview_runner.preview_model_result
        ):
            # select result
            ind = np.flatnonzero(self.curveplot.selection_group)[0]
            row_id = self.curveplot.data[ind].id
            result = self.preview_runner.preview_model_result.get(row_id, None)
        if result is not None:
            # show total fit
            eval = np.atleast_2d(np.broadcast_to(result.eval(x=x), x.shape))
            di_abs = [("curve", (x, np.abs(eval), INTEGRATE_DRAW_CURVE_PENARGS))]
            dis_abs.append({"draw": di_abs, "color": 'red'})
            di_angle = [("curve", (x, np.angle(eval), INTEGRATE_DRAW_CURVE_PENARGS))]
            dis_angle.append({"draw": di_angle, "color": 'red'})
            # show components
            eval_comps = result.eval_components(x=x)
            for i in range(self.preprocessormodel.rowCount()):
                item = self.preprocessormodel.item(i)
                prefix = unique_prefix(item.data(DescriptionRole).viewclass, i)
                comp = eval_comps.get(prefix, None)
                if comp is not None:
                    comp = np.atleast_2d(np.broadcast_to(comp, x.shape))
                    di_abs = [
                        ("curve", (x, np.abs(comp), INTEGRATE_DRAW_CURVE_PENARGS))
                    ]
                    di_angle = [
                        ("curve", (x, np.angle(comp), INTEGRATE_DRAW_CURVE_PENARGS))
                    ]
                    color = self.flow_view.preview_color(i)
                    dis_abs.append({"draw": di_abs, "color": color})
                    dis_angle.append({"draw": di_angle, "color": color})

        refresh_integral_markings(dis_abs, self.markings_list, self.curveplot)
        refresh_integral_markings(
            dis_angle, self.markings_list_after, self.curveplot_after
        )


def create_model_list(m_def: list[None]) -> tuple[list[Model], Parameters]:
    """create_composite_model() but returns list of models instead"""
    # TODO move to owpeakfit, split create_composite_model up
    n = len(m_def)
    m_list = []
    parameters = Parameters()
    for i in range(n):
        item = m_def[i]
        m = create_model(item, i)
        p = prepare_params(item, m)
        m_list.append(m)
        parameters.update(p)

    model = None
    if m_list:
        model = m_list

    return model, parameters


def add_editor(cls, widget):
    widget.add_preprocessor(pack_model_editor(cls))
    editor = widget.flow_view.widgets()[-1]
    return editor


def add_fixed_params(editor, params: dict):
    for k, v in params.items():
        editor.set_hint(k, 'value', v)
        editor.set_hint(k, 'vary', 'fixed')


if __name__ == "__main__":  # pragma: no cover
    data = Cut(lowlim=1680, highlim=1800)(Table("collagen")[0:1])
    wp = WidgetPreview(OWSnomModel)
    wp.run(data, no_exec=True, no_exit=True)
    # Demo PMMA model
    # Air
    add_fixed_params(
        add_editor(StaticPermittivityEditor, wp.widget),
        {
            'c': 1,
        },
    )
    # PMMA
    add_fixed_params(
        add_editor(LorentzianPermittivityEditor, wp.widget),
        {
            'nu_j': 1738,
            'A_j': 100000,
            'gamma_j': 20,
            'eps_inf': 2,
        },
    )
    # Si permitivitty in the mid-infrared
    add_fixed_params(
        add_editor(StaticPermittivityEditor, wp.widget),
        {
            'c': 11.7,
        },
    )
    wp.widget.show_preview(show_info_anyway=True)
    # Rest of run()
    exit_code = wp.exec_widget()
    wp.tear_down()
    sys.exit(exit_code)
