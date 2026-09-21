from __future__ import annotations

from typing import Any, cast

from cellonaut.gui.state import PreviewState  # noqa: E402
from cellonaut.gui.results import CellonautGuiResultsMixin
from cellonaut.gui.preview_loading import CellonautGuiPreviewLoadingMixin


class PreviewFallbackHarness(CellonautGuiResultsMixin):
    def __init__(self):
        self.preview_state = PreviewState()
        self.preview_state.pages = cast(Any, [object()])
        self.preview_state.page_index = 0
        self.logs: list[str] = []
        self.rendered = None

    def set_preview_layer_items_for_page(self, _index: int, *, preserve_view: bool = False):
        raise RuntimeError("layer renderer failed")

    def refresh_tiff_preview_page(self, _index: int, *, preserve_view: bool = False):
        raise ValueError("TIFF renderer failed")

    def set_preview_pixmap(self, pixmap, preserve_view: bool = False):
        self.rendered = (pixmap, preserve_view)

    def update_preview_page_controls(self):
        pass

    def log(self, message: str):
        self.logs.append(message)


def test_preview_rendering_fallback_reports_failed_renderers():
    preview = PreviewFallbackHarness()

    preview.show_preview_page(0, preserve_view=True)

    assert preview.rendered == (preview.preview_state.pages[0], True)
    assert preview.logs == [
        "[PREVIEW][WARN] Layered page rendering failed: RuntimeError: layer renderer failed",
        "[PREVIEW][WARN] TIFF page rendering failed: ValueError: TIFF renderer failed",
    ]


class PreviewOpenFailureHarness(CellonautGuiPreviewLoadingMixin):
    def __init__(self):
        self.preview_state = PreviewState()
        self._current_open_right_panel_path = "broken.tif"
        self.preview_info_label = cast(Any, type("Label", (), {"setText": lambda self, value: None})())
        self.messages = []

    def disable_preview_snapshot(self):
        pass

    def preview_tiff_image(self, file_path):
        _ = file_path
        raise ValueError("bad TIFF")

    def reset_preview_display(self, **_kwargs):
        pass

    def update_preview_tools_context(self):
        pass

    def log(self, message):
        self.messages.append(message)


def test_failed_preview_open_clears_path_guard_and_logs_error():
    preview = PreviewOpenFailureHarness()

    assert preview.preview_file("broken.tif") is False
    assert preview._current_open_right_panel_path is None
    assert preview.messages == ["[PREVIEW][ERROR] Could not preview file: broken.tif (ValueError: bad TIFF)"]
