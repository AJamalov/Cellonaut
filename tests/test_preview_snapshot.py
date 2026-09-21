from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect, QRectF  # noqa: E402
from PySide6.QtTest import QTest
from PySide6.QtGui import QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QGraphicsView  # noqa: E402

from cellonaut.gui.main_window import CellonautMainWindow  # noqa: E402
from cellonaut.pipeline.processing_montage import save_montage_png  # noqa: E402


pytestmark = pytest.mark.gui


def make_gui() -> CellonautMainWindow:
    app = QApplication.instance() or QApplication([])
    _ = app
    return CellonautMainWindow()


def test_switching_images_restores_panning_after_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(CellonautMainWindow, "start_pending_fiji_component_scan", lambda self: None)
    preview = make_gui()
    image = QImage(64, 64, QImage.Format.Format_RGB32)
    image.fill(0x00FF00)
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    assert image.save(str(first))
    assert image.save(str(second))
    preview.preview_file(str(first))
    preview.preview_snapshot_toggle_button.setChecked(True)
    assert preview.preview_view.dragMode() == QGraphicsView.DragMode.NoDrag

    preview.preview_file(str(second))

    assert not preview.preview_snapshot_toggle_button.isChecked()
    assert preview._preview_snapshot_item is None
    assert preview.preview_view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
    preview.preview_snapshot_toggle_button.setChecked(True)
    assert preview.preview_view.dragMode() == QGraphicsView.DragMode.NoDrag

def test_switching_images_preserves_zoom_and_position(tmp_path, monkeypatch):
    monkeypatch.setattr(CellonautMainWindow, "start_pending_fiji_component_scan", lambda self: None)
    preview = make_gui()
    preview.show()
    QApplication.processEvents()
    image = QImage(1600, 1200, QImage.Format.Format_RGB32)
    image.fill(0x00FF00)
    first, second = tmp_path / "zoom_first.png", tmp_path / "zoom_second.png"
    assert image.save(str(first))
    assert image.save(str(second))
    preview.preview_file(str(first))
    for _ in range(10):
        preview.preview_view.zoom_by_steps(1)
    preview.preview_view.centerOn(1100, 800)
    QApplication.processEvents()
    QTest.mouseMove(preview.preview_view.viewport(), QPoint(30, 40))
    before = preview.capture_preview_view_state()
    percent = preview.preview_view.zoom_percent()
    assert preview.preview_file(str(second))
    QApplication.processEvents()
    after = preview.capture_preview_view_state()
    assert after == before
    for path in [first, second] * 5:
        assert preview.preview_file(str(path))
        QApplication.processEvents()
        assert preview.capture_preview_view_state() == before
    assert preview.preview_view.zoom_percent() == percent
    preview.preview_view.fit_image()
    assert preview.preview_view.zoom_percent() == 100




def test_visible_snapshot_saves_square_crop_from_preview(tmp_path: Path):
    preview = make_gui()
    preview.output_dir.set(str(tmp_path / "run_output"))
    source_path = tmp_path / "green.png"
    image = QImage(8, 8, QImage.Format.Format_RGB32)
    image.fill(0x00FF00)
    assert image.save(str(source_path))
    preview.preview_file(str(source_path))
    preview.preview_snapshot_toggle_button.setChecked(True)
    assert preview._preview_snapshot_item is not None
    preview._preview_snapshot_item.setRect(QRectF(2, 2, 4, 4))

    preview.save_visible_snapshot_from_selection()
    out_path = tmp_path / "run_output" / "Results" / "Image Preview Tools" / "Snapshots" / "green_snapshot.png"

    assert out_path.exists()
    result = QImage(str(out_path))
    assert result.width() == 4
    assert result.height() == 4
    assert result.pixelColor(1, 1).green() > 200


def test_layer_snapshot_montage_saves_crop_for_each_layer(tmp_path: Path):
    preview = make_gui()
    data = np.zeros((1, 1, 3, 8, 8), dtype=np.uint8)
    data[0, 0, 0, :, :] = 40
    data[0, 0, 1, 1:7, 1:7] = 180
    data[0, 0, 2, 3:8, 3:8] = 255
    preview.preview_state.tiff_model = {"data": data, "is_overlay": True, "source_shape": data.shape, "dtype": "uint8"}
    preview.preview_state.current_labels = ["Base", "Mask A", "Mask B"]
    preview.preview_state.current_layer_roles = ["image", "weka_mask", "weka_mask"]
    preview.preview_state.current_layer_keys = ["base", "mask_a", "mask_b"]
    preview.preview_state.overlay_colors = ["#FFFFFF", "#FF00FF", "#00FFFF"]
    preview.preview_state.overlay_opacities = [1.0, 1.0, 1.0]
    preview.preview_state.page_index = 0
    preview.preview_scene.setSceneRect(0, 0, 8, 8)
    preview.update_preview_snapshot_controls()
    assert preview.preview_snapshot_montage_action.isEnabled()
    preview.preview_snapshot_toggle_button.setChecked(True)
    assert preview._preview_snapshot_item is not None
    preview._preview_snapshot_item.setRect(QRectF(2, 2, 4, 4))

    crops = preview.snapshot_layer_crops()
    assert [label for label, _pixmap in crops] == ["Base", "Mask A", "Mask B", "Overlay"]

    out_path = tmp_path / "layers_montage.png"
    saved = preview.save_layer_snapshot_montage(out_path)

    assert saved == out_path
    result = QImage(str(out_path))
    assert result.width() >= 20
    assert result.height() >= 28


def test_flat_image_disables_layer_montage_and_file_change_clears_snapshot(tmp_path: Path):
    preview = make_gui()
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first = QImage(8, 8, QImage.Format.Format_RGB32)
    first.fill(0xFF0000)
    second = QImage(8, 8, QImage.Format.Format_RGB32)
    second.fill(0x0000FF)
    assert first.save(str(first_path))
    assert second.save(str(second_path))

    preview.preview_file(str(first_path))
    assert not preview.preview_snapshot_montage_action.isEnabled()
    preview.preview_snapshot_toggle_button.setChecked(True)
    assert preview.preview_snapshot_enabled()

    preview.preview_file(str(second_path))

    assert not preview.preview_snapshot_enabled()
    assert not preview.preview_snapshot_capture_button.isEnabled()
    assert not preview.preview_snapshot_montage_action.isEnabled()


def test_processing_montage_snapshot_repeats_position_in_panel_order(tmp_path: Path):
    preview = make_gui()
    montage_path = tmp_path / "processing_montage.png"
    tiles = [
        ("Raw", np.arange(64, dtype=np.uint8).reshape(8, 8)),
        ("Processed", np.arange(64, dtype=np.uint8).reshape(8, 8) + 32),
        ("Mask", np.eye(8, dtype=np.uint8) * 255),
    ]
    assert save_montage_png(tiles, montage_path, max_tile_side=8, columns=2) == montage_path

    preview.preview_file(str(montage_path))
    assert preview.preview_snapshot_montage_action.isEnabled()
    layout = preview.processing_montage_snapshot_layout()
    first_tile = layout["tiles"][0]
    preview.preview_snapshot_toggle_button.setChecked(True)
    assert preview._preview_snapshot_item is not None
    preview._preview_snapshot_item.setRect(
        QRectF(float(first_tile["x"] + 2), float(first_tile["y"] + 1), 3.0, 3.0)
    )

    crops = preview.snapshot_layer_crops()

    assert [label for label, _pixmap in crops] == ["Raw", "Processed", "Mask"]
    source = QPixmap(str(montage_path))
    for (_label, crop), tile in zip(crops, layout["tiles"]):
        expected = source.copy(QRect(int(tile["x"]) + 2, int(tile["y"]) + 1, 3, 3))
        assert crop.toImage() == expected.toImage()


def test_full_preview_export_preserves_rectangular_image_and_visible_group_layer(tmp_path, monkeypatch):
    from PIL import Image
    from PySide6.QtWidgets import QGraphicsPixmapItem

    monkeypatch.setattr(CellonautMainWindow, "start_pending_fiji_component_scan", lambda self: None)
    preview = make_gui()
    source = QImage(12, 8, QImage.Format.Format_RGB32)
    source.fill(0x000000)
    source_path = tmp_path / "source.png"
    assert source.save(str(source_path))
    preview.preview_file(str(source_path))
    assert preview.preview_export_image_button.isEnabled()
    group_image = QImage(4, 3, QImage.Format.Format_ARGB32)
    group_image.fill(0xFFFF0000)
    group = QGraphicsPixmapItem(QPixmap.fromImage(group_image))
    group.setPos(2, 2)
    group.setZValue(100)
    group.setOpacity(0.5)
    preview.preview_scene.addItem(group)
    preview.preview_snapshot_toggle_button.setChecked(True)
    snapshot_item = preview._preview_snapshot_item
    assert snapshot_item is not None
    snapshot_item.setRect(QRectF(0, 0, 3, 3))
    preview.preview_view.zoom_by_steps(2)

    for suffix in ("png", "tif"):
        destination = tmp_path / f"edited.{suffix}"
        assert preview.save_preview_image(destination) == destination
        with Image.open(destination) as exported:
            assert exported.size == (12, 8)
            assert exported.mode == "RGB"
            pixel = exported.getpixel((3, 3))
            assert isinstance(pixel, tuple) and len(pixel) == 3
            red, green, blue = pixel
            assert 120 <= red <= 135
            assert green == blue == 0
            assert exported.getpixel((10, 6)) == (0, 0, 0)
        assert snapshot_item.isVisible()
    group.hide()
    destination = tmp_path / "hidden.png"
    preview.save_preview_image(destination)
    with Image.open(destination) as exported:
        assert exported.getpixel((3, 3)) == (0, 0, 0)
    preview.close()
