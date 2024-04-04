from Orange.widgets.widget import OWWidget
from Orange.widgets.gui import label


class OWHelloSpectroscopy(OWWidget):
    name = "Hello Spectroscopy!"
    want_main_area = False

    category = "Spectroscopy"

    def __init__(self):
        super().__init__()
        label(self.controlArea, self, "Hello")


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWHelloSpectroscopy).run()
