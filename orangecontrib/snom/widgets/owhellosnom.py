from Orange.widgets.widget import OWWidget
from Orange.widgets.gui import label


class OWHelloSNOM(OWWidget):
    name = "Hello SNOM!"
    want_main_area = False

    def __init__(self):
        super().__init__()
        label(self.controlArea, self, "Hello")


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWHelloSNOM).run()
