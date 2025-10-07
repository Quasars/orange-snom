import sys

import numpy as np
import snompy.sample
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.data.owpreprocess import PreprocessAction, Description, icon_path
from Orange.widgets.data.utils.preprocess import DescriptionRole
from lmfit import Model
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
from orangecontrib.spectroscopy.widgets.peak_editors import ModelEditor


# Wrapping existing function, so re-using "A_j" notation (for now).
def lorentz_perm(x, nu_j=0.0, gamma_j=1.0, A_j=1.0, eps_inf=1.0):  # noqa: N803
    return snompy.sample.lorentz_perm(x, nu_j, gamma_j, A_j=A_j, eps_inf=eps_inf)


class LorentzianPermittivityModel(Model):
    def __init__(
        self, independent_vars=('x',), prefix='', nan_policy='raise', **kwargs
    ):
        kwargs.update(
            {
                'prefix': prefix,
                'nan_policy': nan_policy,
                'independent_vars': independent_vars,
            }
        )
        super().__init__(lorentz_perm, **kwargs)


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


PREPROCESSORS = [pack_model_editor(e) for e in [LorentzianPermittivityEditor]]


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


if __name__ == "__main__":  # pragma: no cover
    data = Cut(lowlim=1680, highlim=1800)(Table("collagen")[0:1])
    wp = WidgetPreview(OWSnomModel)
    wp.run(data, no_exec=True, no_exit=True)
    # Demo PMMA model
    p = pack_model_editor(LorentzianPermittivityEditor)
    wp.widget.add_preprocessor(p)
    editor = wp.widget.flow_view.widgets()[-1]
    editor.set_hint('nu_j', 'value', 1738)
    editor.set_hint('A_j', 'value', 100000)
    editor.set_hint('gamma_j', 'value', 20)
    editor.set_hint('eps_inf', 'value', 2)
    editor.set_hint('eps_inf', 'vary', 'fixed')
    wp.widget.show_preview(show_info_anyway=True)
    # Rest of run()
    exit_code = wp.exec_widget()
    wp.tear_down()
    sys.exit(exit_code)
