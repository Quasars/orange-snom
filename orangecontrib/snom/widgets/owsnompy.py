import multiprocessing
import concurrent.futures
import sys
import time

import Orange
import numpy as np
from Orange.data import Table, Domain
from Orange.widgets import gui
from Orange.widgets.data.owpreprocess import PreprocessAction, Description, icon_path
from Orange.widgets.data.utils.preprocess import DescriptionRole
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.utils.sql import check_sql_input

from lmfit.model import ModelResult
from orangecontrib.spectroscopy.preprocess.utils import replacex
from orangecontrib.spectroscopy.widgets.owpreprocess import (
    InterruptException,
    SpectralPreprocess,
)
from orangecontrib.spectroscopy.widgets.owspectra import SELECTONE
from orangewidget.utils.signals import Input

from orangecontrib.snom.widgets.snompy_compute import (
    pool_fit2,
    pool_initializer,
    pool_fit,
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
    create_model,
    prepare_params,
    PeakPreviewRunner,
    unique_prefix,
    N_PROCESSES,
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
    EffPolNFdmParams,
    SigmaNParams,
    SigmaN,
)
from orangecontrib.snom.widgets.snompy_util import (
    create_model_list,
    load_list,
    fit_results_table,
    load_op,
)
from orangecontrib.snom.temp import FitPreprocess, ComplexTable


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
        m_def = master.save(master.preprocessormodel)
        if master.data is not None:
            # Clear markings to indicate preview is running
            refresh_integral_markings(
                [], master.markings_list, master.curveplot_amplitude
            )
            refresh_integral_markings(
                [], master.markings_list_after, master.curveplot_phase
            )
            data = master.sample_data(master.data)
            # Pass preview data to widgets here as we don't use on_partial_result()
            for w in self.master.flow_view.widgets():
                w.set_preview_data(data)
            self.start(self.run_preview, data, m_def, self.pool)
        else:
            master.curveplot_amplitude.set_data(None)
            master.curveplot_phase.set_data(None)

    def on_done(self, result):
        orig_data, after_data, model_result = result
        final_preview = self.preview_pos is None
        if final_preview:
            self.preview_data = orig_data
            self.after_data = orig_data

        if self.preview_data is None:  # happens in OWIntegrate
            self.preview_data = orig_data

        self.preview_model_result = model_result

        # Plot complex data as amplitude and phase
        if isinstance(self.preview_data, ComplexTable):
            self.master.curveplot_amplitude.set_data(
                self.preview_data.to_amplitude_table()
            )
            self.master.curveplot_phase.set_data(self.preview_data.to_phase_table())
        else:
            self.master.curveplot_amplitude.set_data(self.preview_data)
            self.master.curveplot_phase.set_data(self.preview_data)

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

        model_result = {}
        x = getx(data)
        if data is not None and m_def is not None and len(m_def['preprocessors']) != 0:
            model_list, _ = create_model_list(load_list(m_def))
            model = compose_model(model_list, load_op(m_def))
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


class OWSnomModel(FitPreprocess):
    name = "SNOM Model"
    description = "Model SNOM spectra with snompy"
    icon = "icons/snompy.svg"

    PREPROCESSORS = PREPROCESSORS
    BUTTON_ADD_LABEL = "Add term..."

    class Inputs:
        data = Input("Data", Orange.data.Table, default=False)
        amplitude = Input("Amplitude", Orange.data.Table, default=False)
        phase = Input("Phase", Orange.data.Table, default=False)

    def __init__(self):
        self.markings_list = []
        self.data_input = None
        self.data_amplitude = None
        self.data_phase = None
        SpectralPreprocess.__init__(self)
        self.curveplot.selection_type = SELECTONE
        self.curveplot.select_at_least_1 = True
        self.curveplot.view_average_menu.setEnabled(False)
        self.curveplot.selection_changed.connect(self.redraw_integral)

        # Show _after (work-around for complex plotting)
        self.curveplot_after.show()
        # Markings list for _after
        self.markings_list_after = []
        # Curveplot aliases
        self.curveplot_amplitude = self.curveplot
        self.curveplot_phase = self.curveplot_after

        # Custom preview running just to plot complex values
        self.preview_runner = ComplexPeakPreviewRunner(self)
        self.preview_runner.preview_updated.connect(self.redraw_integral)

        # Model options
        gui.widgetBox(self.controlArea, "Model Options")

    @Inputs.data
    @check_sql_input
    def set_data_input(self, data=None):
        """Set the input data set."""
        self.data_input = data

    @Inputs.amplitude
    @check_sql_input
    def set_data_amplitude(self, data=None):
        """Set the input data set."""
        self.data_amplitude = data

    @Inputs.phase
    @check_sql_input
    def set_data_phase(self, data=None):
        """Set the input data set."""
        self.data_phase = data

    def handleNewSignals(self):
        if self.data_amplitude and self.data_phase:
            self.data = ComplexTable.from_amplitude_phase_tables(
                self.data_amplitude, self.data_phase
            )
        elif self.data_input:
            if isinstance(self.data_input, ComplexTable):
                self.data = self.data_input
            elif "channel" in self.data_input.domain:
                self.data = ComplexTable.from_interleaved_table(self.data_input)
            else:
                self.data = self.data_input
        else:
            self.data = None
        super().handleNewSignals()

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

        refresh_integral_markings(dis_abs, self.markings_list, self.curveplot_amplitude)
        refresh_integral_markings(
            dis_angle, self.markings_list_after, self.curveplot_phase
        )

    def snompy_op(self):
        eff_pol_n_params = EffPolNFdmParams(
            A_tip=20e-9, n=3, r_tip=30e-9, L_tip=350e-9, method="Q_ave"
        )
        sigma_n_params = SigmaNParams(
            **eff_pol_n_params, theta_in=np.deg2rad(60), c_r=0.3
        )
        return SigmaN(sigma_n_params)

    def save(self, model):
        d = super().save(model)

        op = self.snompy_op()
        snompy_params = op.parameters
        snompy_params["op"] = type(op).__qualname__
        d["snompy"] = snompy_params
        return d

    def create_outputs(self):
        m_def = self.save(self.preprocessormodel)
        self.start(self.run_task, self.data, m_def)

    @staticmethod
    def run_task(data: Table, m_def, state: TaskState):
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

        data_fits = data_anno = data_resid = None
        if data is not None and m_def is not None and len(m_def["preprocessors"]) != 0:
            orig_data = data
            output = []
            x = getx(data)
            n = len(data)
            fits = []
            residuals = []

            with multiprocessing.Pool(
                processes=N_PROCESSES, initializer=pool_initializer, initargs=(m_def, x)
            ) as p:
                res = p.map_async(pool_fit, data.X, chunksize=1)

                def done():
                    try:
                        return n - res._number_left * res._chunksize
                    except AttributeError:
                        return 0

                while not res.ready():
                    progress_interrupt(done() / n * 99)
                    res.wait(0.05)

                fitsr = res.get()

            progress_interrupt(99)
            progress_interrupt(99)

            for mrd, bpar, fitted, resid in fitsr:
                model_results_dict = mrd
                output.append(bpar)
                fits.append(fitted)
                residuals.append(resid)
                progress_interrupt(99)
            data = fit_results_table(np.vstack(output), model_results_dict, orig_data)
            data_fits = orig_data.from_table_rows(orig_data, ...)  # a shallow copy
            with data_fits.unlocked_reference(data_fits.X):
                data_fits.X = np.vstack(fits)
            data_resid = orig_data.from_table_rows(orig_data, ...)  # a shallow copy
            with data_resid.unlocked_reference(data_resid.X):
                data_resid.X = np.vstack(residuals)
            dom_anno = Domain(
                orig_data.domain.attributes,
                orig_data.domain.class_vars,
                orig_data.domain.metas + data.domain.attributes,
            )
            data_anno = orig_data.transform(dom_anno)
            with data_anno.unlocked(data_anno.metas):
                data_anno.metas[:, len(orig_data.domain.metas) :] = data.X

        progress_interrupt(100)

        return data, data_fits, data_resid, data_anno


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
    wp.run(set_data_input=data, no_exec=True, no_exit=True)
    # Demo PMMA model
    demo_pmma_model(wp.widget)
    wp.widget.show_preview(show_info_anyway=True)
    wp.widget.commit.deferred()
    # Rest of run()
    exit_code = wp.exec_widget()
    wp.tear_down()
    sys.exit(exit_code)
