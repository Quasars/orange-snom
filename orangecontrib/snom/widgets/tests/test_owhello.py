from Orange.widgets.tests.base import WidgetTest

from orangecontrib.snom.widgets.owhellosnom import OWHelloSNOM
from orangecontrib.snom.widgets.owhellospectroscopy import OWHelloSpectroscopy


class TestOWHelloSNOM(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWHelloSNOM)

    def test_activation(self):
        pass


class TestOWHelloSpectroscopy(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWHelloSpectroscopy)

    def test_activation(self):
        pass
