import numpy as np

from orangewidget.tests.utils import excepthook_catch

from Orange.data import Table, Domain
from Orange.widgets.tests.base import WidgetTest
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.tests import spectral_preprocess
from orangecontrib.spectroscopy.tests.spectral_preprocess import (
    pack_editor,
    wait_for_preview,
)
from orangecontrib.spectroscopy.widgets.preprocessors.misc import (
    CutEditor,
    SavitzkyGolayFilteringEditor,
)

from orangecontrib.snom.widgets.owpreprocessimage import OWPreprocessImage
from orangecontrib.snom.widgets.preprocessors.registry import preprocess_image_editors


PREPROCESSORS = list(map(pack_editor, preprocess_image_editors.sorted()))


WHITELIGHT = Table("whitelight.gsf")


class TestAllPreprocessors(WidgetTest):
    def test_allpreproc_indv(self):
        data = WHITELIGHT
        for p in PREPROCESSORS:
            with self.subTest(p.viewclass):
                self.widget = self.create_widget(OWPreprocessImage)
                self.send_signal("Data", data)
                self.widget.add_preprocessor(p)
                self.widget.commit.now()
                wait_for_preview(self.widget)
                self.wait_until_finished(timeout=10000)

    def test_allpreproc_indv_empty(self):
        data = WHITELIGHT
        for p in PREPROCESSORS:
            with self.subTest(p.viewclass):
                self.widget = self.create_widget(OWPreprocessImage)
                self.send_signal("Data", data[:0])
                self.widget.add_preprocessor(p)
                self.widget.commit.now()
                wait_for_preview(self.widget)
                self.wait_until_finished(timeout=10000)
        # no attributes
        data = data.transform(
            Domain([], class_vars=data.domain.class_vars, metas=data.domain.metas)
        )
        for p in PREPROCESSORS:
            with self.subTest(p.viewclass, type="no attributes"):
                self.widget = self.create_widget(OWPreprocessImage)
                self.send_signal("Data", data)
                self.widget.add_preprocessor(p)
                self.widget.commit.now()
                wait_for_preview(self.widget)
                self.wait_until_finished(timeout=10000)


class TestOWPreprocess(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocessImage)

    def test_load_unload(self):
        self.send_signal("Data", Table("iris.tab"))
        self.send_signal("Data", None)

    def failing_saving_preprocessors(self):
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.assertEqual([], settings["storedsettings"]["preprocessors"])
        self.widget.add_preprocessor(self.widget.PREPROCESSORS[0])
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.assertEqual(
            self.widget.PREPROCESSORS[0].qualname,
            settings["storedsettings"]["preprocessors"][0][0],
        )

    def test_output_preprocessor_without_data(self):
        self.widget.add_preprocessor(pack_editor(CutEditor))
        self.widget.commit.now()
        self.wait_until_finished()
        out = self.get_output(self.widget.Outputs.preprocessor)
        self.assertIsInstance(out, Preprocess)

    def test_empty_no_inputs(self):
        self.widget.commit.now()
        self.wait_until_finished()
        p = self.get_output(self.widget.Outputs.preprocessor)
        d = self.get_output(self.widget.Outputs.preprocessed_data)
        self.assertEqual(None, p)
        self.assertEqual(None, d)

    def test_no_preprocessors(self):
        data = WHITELIGHT
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.commit.now()
        self.wait_until_finished()
        d = self.get_output(self.widget.Outputs.preprocessed_data)
        self.assertEqual(WHITELIGHT, d)

    def test_widget_vs_manual(self):
        data = WHITELIGHT
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.add_preprocessor(pack_editor(CutEditor))
        self.widget.add_preprocessor(pack_editor(SavitzkyGolayFilteringEditor))
        self.widget.commit.now()
        self.wait_until_finished()
        p = self.get_output(self.widget.Outputs.preprocessor)
        d = self.get_output(self.widget.Outputs.preprocessed_data)
        manual = p(data)
        np.testing.assert_equal(d.X, manual.X)

    def test_invalid_preprocessors(self):
        settings = {"storedsettings": {"preprocessors": [("xyz.abc.notme", {})]}}
        with self.assertRaises(KeyError):
            with excepthook_catch(raise_on_exit=True):
                widget = self.create_widget(OWPreprocessImage, settings)
                self.assertTrue(widget.Error.loading.is_shown())


class TestPreprocessWarning(spectral_preprocess.TestWarning):
    widget_cls = OWPreprocessImage

    def test_exception_preview_after_data(self):
        self.editor.raise_exception = True
        self.editor.edited.emit()
        wait_for_preview(self.widget)
        self.assertIsNone(self.widget.curveplot_after.data)

        self.editor.raise_exception = False
        self.editor.edited.emit()
        wait_for_preview(self.widget)
        self.assertIsNotNone(self.widget.curveplot_after.data)
