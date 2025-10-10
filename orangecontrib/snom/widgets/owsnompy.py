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

from lmfit.model import ModelResult
from orangecontrib.spectroscopy.preprocess.utils import replacex
from orangecontrib.spectroscopy.widgets.owpreprocess import InterruptException

from orangecontrib.snom.widgets.snompy_compute import (
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
    FiniteInterface,
    Reference,
    Interface,
)
from orangecontrib.snom.widgets.snompy_util import create_model_list, load_list


class StaticPermittivityEditor(ConstantModelEditor):
    name = "Static Permittivity"


class PlaceholderEditor(ModelEditor):
    category = "Placeholder"
    icon = "Continuize.svg"

    @classmethod
    def createinstance(cls, prefix, form=None):
        return cls.model()

    @staticmethod
    def model_parameters():
        return ()

    @staticmethod
    def model_lines():
        return ()


class InterfaceEditor(PlaceholderEditor):
    name = "Interface"
    model = Interface
    prefix_generic = 'if'


class FiniteInterfaceEditor(ConstantModelEditor):
    name = "Finite Interface"
    model = FiniteInterface
    prefix_generic = 'fif'


class ReferenceEditor(PlaceholderEditor):
    name = "Reference"
    model = Reference
    prefix_generic = 'ref'


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

    @staticmethod
    def model_lines():
        return ()


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
        DrudePermittivityEditor,
        StaticPermittivityEditor,
        InterfaceEditor,
        FiniteInterfaceEditor,
        ReferenceEditor,
    ]
]


class ComplexPeakPreviewRunner(PeakPreviewRunner):
    def show_preview(self, show_info_anyway=False):
        """Shows preview and also passes preview data to the widgets
        Re-implmented to change how pp_def / m_def is generated
        """
        master = self.master
        self.preview_pos = master.flow_view.preview_n()
        self.last_partial = None
        self.show_info_anyway = show_info_anyway
        self.preview_data = None
        self.after_data = None
        pp_def = master.save(master.preprocessormodel)
        if master.data is not None:
            # Clear markings to indicate preview is running
            refresh_integral_markings([], master.markings_list, master.curveplot)
            data = master.sample_data(master.data)
            # Pass preview data to widgets here as we don't use on_partial_result()
            for w in self.master.flow_view.widgets():
                w.set_preview_data(data)
            self.start(self.run_preview, data, pp_def, self.pool)
        else:
            master.curveplot.set_data(None)
            master.curveplot_after.set_data(None)

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

        model_list, parameters = create_model_list(load_list(m_def))
        model = compose_model(model_list)

        model_result = {}
        x = getx(data)
        if data is not None and len(m_def) != 0:
            for row in data:
                progress_interrupt(0)
                res = pool.schedule(pool_fit2, (row.x, m_def, x))
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
                model_result[row.id] = ModelResult(model, fits)

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


def demo_pmma_model(widget):
    """Demo PMMA model from t_dependent_spectra.py"""
    editors = {
        'name': "",
        'preprocessors': [
            # Air
            (
                'orangecontrib.spectroscopy.widgets.owsnompy.const',
                {
                    'c': {'value': 1, 'vary': 'fixed'},
                },
            ),
            # PMMA
            (
                'orangecontrib.spectroscopy.widgets.owsnompy.fif',
                {'c': {'value': 35 * 1e-9, 'vary': 'fixed'}},
            ),
            (
                'orangecontrib.spectroscopy.widgets.owsnompy.lp',
                {
                    'nu_j': {'value': 1738e2, 'vary': 'fixed'},
                    'A_j': {'value': 4.2e8, 'vary': 'fixed'},
                    'gamma_j': {'value': 20e2, 'vary': 'fixed'},
                    'eps_inf': {'value': 2, 'vary': 'fixed'},
                },
            ),
            # Si permitivitty in the mid-infrared
            ('orangecontrib.spectroscopy.widgets.owsnompy.if', {}),
            (
                'orangecontrib.spectroscopy.widgets.owsnompy.const',
                {
                    'c': {'value': 11.7, 'vary': 'fixed'},
                },
            ),
            # Au reference
            # Air
            ('orangecontrib.spectroscopy.widgets.owsnompy.ref', {}),
            (
                'orangecontrib.spectroscopy.widgets.owsnompy.const',
                {'c': {'value': 1, 'vary': 'fixed'}},
            ),
            # Au
            ('orangecontrib.spectroscopy.widgets.owsnompy.if', {}),
            (
                'orangecontrib.spectroscopy.widgets.owsnompy.dp',
                {
                    'nu_plasma': {'value': 7.25e6, 'vary': 'fixed'},
                    'gamma': {'value': 2.16e4, 'vary': 'fixed'},
                    'eps_inf': {'value': 1, 'vary': 'fixed'},
                },
            ),
        ],
    }
    widget.set_model(widget.load(editors))


if __name__ == "__main__":  # pragma: no cover
    data = Cut(lowlim=1680, highlim=1800)(Table("collagen")[0:1])
    new_x = getx(data) * 100
    data = replacex(data, new_x)
    wp = WidgetPreview(OWSnomModel)
    wp.run(data, no_exec=True, no_exit=True)
    # Demo PMMA model
    demo_pmma_model(wp.widget)
    wp.widget.show_preview(show_info_anyway=True)
    # Rest of run()
    exit_code = wp.exec_widget()
    wp.tear_down()
    sys.exit(exit_code)
