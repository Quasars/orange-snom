from Orange.data import Table
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.concurrent import TaskState
from orangecontrib.spectroscopy.widgets.owpreprocess import SpectralPreprocess
from orangewidget.utils.signals import Output


# TODO move to orangecontrib.spectroscopy
class FitPreprocess(SpectralPreprocess, openclass=True):
    BUTTON_ADD_LABEL = "Add model..."

    class Outputs:
        fit_params = Output("Fit Parameters", Table, default=True)
        fits = Output("Fits", Table)
        residuals = Output("Residuals", Table)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    preview_on_image = True

    def __init__(self):
        super().__init__()

    def redraw_integral(self):
        """Widget-specific"""
        raise NotImplementedError

    def create_outputs(self):
        """Currently widget-specific (different m_def serializations)"""
        raise NotImplementedError

    @staticmethod
    def run_task(data: Table, m_def, state: TaskState):
        """Currently widget-specific (different m_def serializations)"""
        raise NotImplementedError

    def on_done(self, results):
        fit_params, fits, residuals, annotated_data = results
        self.Outputs.fit_params.send(fit_params)
        self.Outputs.fits.send(fits)
        self.Outputs.residuals.send(residuals)
        self.Outputs.annotated_data.send(annotated_data)

    def on_exception(self, ex):
        try:
            super().on_exception(ex)
        except ValueError:
            self.Error.applying(ex)
            self.Outputs.fit_params.send(None)
            self.Outputs.fits.send(None)
            self.Outputs.residuals.send(None)
            self.Outputs.annotated_data.send(None)
