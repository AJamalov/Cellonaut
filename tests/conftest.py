"""Shared real-file fixtures for integration tests."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile


RELEASE_TEST_MODULES = frozenset(
    {
        "test_corresponding_sources.py",
        "test_dependency_lock.py",
        "test_dependency_versions.py",
        "test_entry_points.py",
        "test_legal_documents.py",
        "test_license_inventory.py",
        "test_offline_assets.py",
        "test_packaged_results.py",
        "test_python_runtime.py",
        "test_qt_bundle_filter.py",
        "test_version.py",
    }
)
INTEGRATION_TEST_MODULES = frozenset(
    {
        "test_cell_group_filter_exports.py",
        "test_imagej_runtime.py",
        "test_pipeline_runtime.py",
        "test_miniature_integration.py",
        "test_scientific_pipeline_contracts.py",
    }
)
SLOW_TEST_MODULES = frozenset(
    {
        "test_gui_startup.py",
        "test_imagej_runtime.py",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply one primary test category and optional runtime traits."""
    for item in items:
        module_name = Path(str(item.path)).name
        if module_name.startswith("test_release_") or module_name in RELEASE_TEST_MODULES:
            item.add_marker(pytest.mark.release)
        elif item.get_closest_marker("gui") is not None or module_name in INTEGRATION_TEST_MODULES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
        if module_name in SLOW_TEST_MODULES:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def qt_application():
    """Keep Qt alive until all test widgets have been destroyed."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def cleanup_test_windows(request: pytest.FixtureRequest):
    """Dispose of windows for tests explicitly marked as GUI tests."""
    if request.node.get_closest_marker("gui") is None:
        yield
        return

    from shiboken6 import delete, isValid
    from cellonaut import resources
    from cellonaut.gui import config
    from cellonaut.gui import build
    from cellonaut.gui import presets
    from cellonaut.gui.main_window import CellonautMainWindow

    app = request.getfixturevalue("qt_application")
    monkeypatch = request.getfixturevalue("monkeypatch")
    tmp_path = request.getfixturevalue("tmp_path")
    existing = set(app.topLevelWidgets())
    app_dir = tmp_path / "Cellonaut"
    monkeypatch.setattr(resources, "USER_APP_DIR", app_dir)
    monkeypatch.setattr(resources, "PRESETS_DIR", app_dir / "presets")
    monkeypatch.setattr(resources, "SETTINGS_FILE", app_dir / "last_settings.json")
    monkeypatch.setattr(presets, "PRESETS_DIR", app_dir / "presets")
    monkeypatch.setattr(build, "USER_APP_DIR", app_dir)
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "last_settings.json")
    yield
    app.processEvents()
    windows = [widget for widget in app.topLevelWidgets() if widget not in existing]
    for window in windows:
        if isValid(window) and isinstance(window, CellonautMainWindow):
            window.close()
            deadline = time.monotonic() + 10
            while window._running_background_threads() and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            running = [
                name
                for name in (
                    "worker_thread",
                    "preview_worker_thread",
                    "nd2_worker_thread",
                    "_nd2_scan_thread",
                    "_input_scan_thread",
                    "_fiji_scan_thread",
                    "_setup_check_thread",
                )
                if (thread := getattr(window, name, None)) is not None and thread.isRunning()
            ]
            assert not running, f"Test left background workers running: {running}"
    app.processEvents()
    for window in windows:
        # Popups also appear in topLevelWidgets; let their parents destroy them.
        if isValid(window) and window.parent() is None:
            delete(window)


@pytest.fixture
def miniature_tiff_dataset(tmp_path: Path) -> dict[str, Any]:
    """Create a tiny, valid two-channel OME-TIFF and matching label/mask TIFFs."""

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    mask_dir = tmp_path / "masks"
    input_dir.mkdir()
    mask_dir.mkdir()

    signal = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 8, 8, 0],
            [0, 2, 6, 2, 0, 8, 12, 0],
            [0, 2, 2, 2, 0, 8, 8, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    reference = np.arange(48, dtype=np.uint16).reshape(6, 8)
    stack = np.stack([signal, reference])
    stack_path = input_dir / "miniature_sample.ome.tif"
    tifffile.imwrite(
        stack_path,
        stack,
        ome=True,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": ["Signal", "Reference"]}},
    )

    cell_labels = np.zeros((6, 8), dtype=np.uint16)
    cell_labels[1:4, 1:4] = 1
    cell_labels[1:4, 5:7] = 2
    cell_labels_path = mask_dir / "miniature_cells.tif"
    tifffile.imwrite(cell_labels_path, cell_labels, photometric="minisblack")

    positive_mask = np.zeros((6, 8), dtype=np.uint8)
    positive_mask[1:3, 1:3] = 255
    positive_mask[1:4, 5] = 255
    positive_mask_path = mask_dir / "miniature_positive_mask.tif"
    tifffile.imwrite(positive_mask_path, positive_mask, photometric="minisblack")

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "stack_path": stack_path,
        "cell_labels_path": cell_labels_path,
        "positive_mask_path": positive_mask_path,
        "signal": signal,
        "reference": reference,
        "cell_labels": cell_labels,
        "positive_mask": positive_mask,
    }


@pytest.fixture
def make_sample_context():
    """Construct a typed sample owner with only the resources a test needs."""
    from cellonaut.pipeline.context import SampleProcessingContext

    def make(**resources):
        fields: dict[str, Any] = dict(id_label="sample", result_id="sample", export_dirs={}, file_map={},
                      image_map={}, roi_map={}, roi_measure_img_map={}, skeleton_metrics={},
                      native_numpy_images=True)
        fields.update(resources)
        return SampleProcessingContext(**fields)
    return make


@pytest.fixture
def imagej_measurement_backend(monkeypatch):
    """Record Java image operations while leaving Python recipe/measurement logic real.

    Commands are recorded, not simulated: this is no oracle for Fiji's image
    transforms. The Analyzer supplies whole-image statistics from actual pixels.
    """
    from types import SimpleNamespace

    backend: Any = SimpleNamespace(copies=[], commands=[], measured=[], on_measure=lambda _image: None)

    class Image:
        def __init__(self, pixels):
            self.pixels = np.asarray(pixels).copy()
            self.history = []
            self.closes = 0
            self.calibration = None
            self.roi = None

        def close(self):
            self.closes += 1

        def setRoi(self, roi):
            self.roi = roi

        def killRoi(self):
            self.roi = None

        def getCalibration(self):
            return self.calibration

        def setCalibration(self, calibration):
            self.calibration = calibration

    class Duplicator:
        def run(self, image):
            assert image.closes == 0
            duplicate = Image(image.pixels)
            duplicate.history = list(image.history)
            backend.copies.append((image, duplicate))
            return duplicate

    class IJ:
        @staticmethod
        def run(image, command, options):
            assert image.closes == 0
            assert command in {"Enhance Contrast", "Smooth", "Subtract Background..."}
            image.history.append((command, options))
            backend.commands.append((image, command, options))

    class ResultsTable:
        def __init__(self):
            self.values = {}

        def getValue(self, label, row):
            assert row == 0
            return self.values[label]

    class Calibration:
        def setUnit(self, unit):
            assert unit == "pixel"

    class WholeImageRoi:
        def clone(self):
            return self

    class Analyzer:
        def __init__(self, image, flags, table):
            self.image, self.table = image, table

        def measure(self):
            assert self.image.closes == 0
            assert isinstance(self.image.roi, WholeImageRoi)
            backend.on_measure(self.image)
            backend.measured.append((self.image, tuple(self.image.history)))
            pixels = self.image.pixels
            self.table.values.update(Area=float(pixels.size), Mean=float(pixels.mean()),
                                     RawIntDen=float(pixels.sum()))

    classes = dict(IJ=IJ, Duplicator=Duplicator, Analyzer=Analyzer, ResultsTable=ResultsTable,
                   Calibration=Calibration, Measurements=SimpleNamespace(AREA=1, MEAN=2, INTEGRATED_DENSITY=4))
    for module in ("cellonaut.pipeline.sample_processing", "cellonaut.pipeline.image_processing",
                   "cellonaut.masks.roi_processing"):
        monkeypatch.setattr(f"{module}.get_java_classes", lambda: classes)
    backend.Image = Image
    backend.roi = WholeImageRoi()
    return backend


@pytest.fixture
def pixel_geometry_cells():
    """Independent pixel-grid geometry expectations, including a non-convex cell."""
    import pandas as pd

    labels = np.zeros((7, 17), dtype=np.int32)
    labels[2:5, 1:4] = 3
    labels[2:5, 6:11] = 17
    labels[2:5, 13:16] = 41
    labels[3, 14] = 0
    # The skimage 4-neighbor perimeter estimator traces boundary-pixel centers:
    # a 3x3 square has 8 unit segments, a 3x5 rectangle has 12. Removing the
    # square's center leaves the same 8-pixel boundary and a 9-pixel convex hull.
    # Circularity is 4*pi*A/P**2 (not clipped to 1); solidity is A/hull_area.
    expected = pd.DataFrame({
        "CellID": [3, 17, 41], "Area": [9.0, 15.0, 8.0], "Perimeter": [8.0, 12.0, 8.0],
        "Solidity": [1.0, 1.0, 8 / 9], "Circularity": [9 * np.pi / 16, 5 * np.pi / 12, np.pi / 2],
    })
    return labels, expected
