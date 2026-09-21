import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from cellonaut.gui.widgets import ElidedLabel


pytestmark = pytest.mark.gui


def test_elided_label_preserves_complete_text_and_tooltip():
    app = QApplication.instance() or QApplication([])
    text = "260212_4798_Hmg2-cherry_BFP-Ubc6_Ost1-neon_e_set2_001.ome_GFP_combined_overlay.tif"
    label = ElidedLabel(text)
    label.resize(120, 30)
    label.show()
    app.processEvents()

    assert label.fullText() == text
    assert "…" in label.text()
    assert label.toolTip() == text
    label.setDetailToolTip("C:/complete/path/" + text)
    label.resize(140, 30)
    app.processEvents()
    assert label.toolTip() == "C:/complete/path/" + text
    label.deleteLater()
    app.processEvents()
