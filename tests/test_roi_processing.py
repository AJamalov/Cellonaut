from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cellonaut.masks.roi_processing import (
    _combined_roi_from_sources,
    apply_fiji_analyze_particles_step,
    apply_fiji_binary_step,
    apply_fiji_translate_step,
    apply_ordered_mask_processing_recipe,
    mask_threshold_method_for_image,
    measure_roi_stats,
    roi_from_mask_array,
    subtract_background_copy,
    union_shape_rois,
)
from cellonaut.config.defaults import MASK_PROCESSING_STEP_BINARY_DILATE, MASK_PROCESSING_STEP_BINARY_FILL_HOLES
from cellonaut.config.processing_steps import parse_binary_settings, parse_particle_settings, parse_translate_offsets


class FakeBounds:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class FakeRoi:
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def clone(self):
        return FakeRoi(self.x, self.y)

    def getBounds(self):
        return FakeBounds(self.x, self.y)

    def setLocation(self, x: int, y: int):
        self.x = x
        self.y = y


class FakeShapeRoi:
    def __init__(self):
        self.union_arg = None

    def or_(self, other):
        self.union_arg = other
        return ("union", other)


class BooleanShapeRoi:
    def __init__(self, pixels):
        self.pixels = set(pixels)

    def clone(self):
        return BooleanShapeRoi(self.pixels)

    def or_(self, other):
        return BooleanShapeRoi(self.pixels | other.pixels)

    def and_(self, other):
        return BooleanShapeRoi(self.pixels & other.pixels)

    def xor_(self, other):
        return BooleanShapeRoi(self.pixels ^ other.pixels)


@pytest.mark.parametrize(
    ("settings", "expected"),
    [("1,1,false", (1, 1, False)), ("100,8,true", (100, 8, True))],
)
def test_fiji_binary_options_parser(settings, expected):
    assert parse_binary_settings(settings) == expected


@pytest.mark.parametrize("settings", ["0,1,false", "101,1,false", "1,9,false", "1.5,1,false", "1,1,"])
def test_fiji_binary_options_parser_rejects_invalid_values(settings):
    with pytest.raises(ValueError, match="Fiji Binary Options"):
        parse_binary_settings(settings)


@pytest.mark.parametrize("settings, expected", [("0,0", (0.0, 0.0)), ("-2.5,3", (-2.5, 3.0))])
def test_fiji_translate_offsets_parser(settings, expected):
    assert parse_translate_offsets(settings) == expected


@pytest.mark.parametrize("settings", ["1", "1,2,3", "x,2", "1,", "NaN,0", "inf,0"])
def test_fiji_translate_offsets_parser_rejects_invalid_values(settings):
    with pytest.raises(ValueError, match="Translate"):
        parse_translate_offsets(settings)


def test_fiji_translate_step_runs_command_and_closes_mask(monkeypatch):
    import cellonaut.masks.roi_processing as processing

    calls = []
    mask = SimpleNamespace(close=lambda: calls.append("closed"))
    monkeypatch.setattr(processing, "get_java_classes", lambda: {"IJ": SimpleNamespace(run=lambda *args: calls.append(args))})
    monkeypatch.setattr(processing, "create_mask_from_roi", lambda *_args: mask)
    monkeypatch.setattr(processing, "imageplus_to_numpy_2d", lambda _mask: np.array([[255]], dtype=np.uint8))
    monkeypatch.setattr(processing, "roi_from_mask_array", lambda array: np.asarray(array))

    result = apply_fiji_translate_step("roi", object(), {"translate_offsets": "-2.5,3"})

    assert isinstance(result, np.ndarray)
    assert result[0, 0] == 255
    assert calls == [(mask, "Translate...", "x=-2.5 y=3 interpolation=None"), "closed"]


def test_fiji_binary_step_runs_command_and_restores_preferences(monkeypatch):
    import cellonaut.masks.roi_processing as processing

    calls = []
    prefs = SimpleNamespace(blackBackground=False, padEdges=True)

    class FakeMask:
        def close(self):
            calls.append(("close",))

    mask = FakeMask()
    monkeypatch.setattr(processing, "get_java_classes", lambda: {"IJ": SimpleNamespace(run=lambda *args: calls.append(args))})
    monkeypatch.setattr(processing, "jimport", lambda name: prefs)
    monkeypatch.setattr(processing, "create_mask_from_roi", lambda *_args: mask)
    monkeypatch.setattr(processing, "imageplus_to_numpy_2d", lambda _mask: np.array([[255]], dtype=np.uint8))
    monkeypatch.setattr(processing, "roi_from_mask_array", lambda array: np.asarray(array))

    result = apply_fiji_binary_step("roi", object(), MASK_PROCESSING_STEP_BINARY_DILATE, {"binary_settings": "3,2,false"})

    assert isinstance(result, np.ndarray)
    assert result[0, 0] == 255
    assert calls == [
        (mask, "Options...", "iterations=3 count=2 black"),
        (mask, "Dilate", ""),
        (mask, "Options...", "iterations=1 count=1 black"),
        ("close",),
    ]
    assert prefs.blackBackground is False
    assert prefs.padEdges is True


def test_fiji_fill_holes_restores_preferences_after_failure(monkeypatch):
    import cellonaut.masks.roi_processing as processing

    closed = []
    prefs = SimpleNamespace(blackBackground=False, padEdges=False)
    mask = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(processing, "get_java_classes", lambda: {"IJ": SimpleNamespace(run=lambda *_args: (_ for _ in ()).throw(RuntimeError("Fiji failed")))})
    monkeypatch.setattr(processing, "jimport", lambda name: prefs)
    monkeypatch.setattr(processing, "create_mask_from_roi", lambda *_args: mask)

    with pytest.raises(RuntimeError, match="Fiji failed"):
        apply_fiji_binary_step("roi", object(), MASK_PROCESSING_STEP_BINARY_FILL_HOLES, {})

    assert prefs.blackBackground is False
    assert prefs.padEdges is False
    assert closed == [True]


def test_fiji_particle_settings_parser():
    assert parse_particle_settings("20-Infinity,0.2-1,true,false") == (20, float("inf"), 0.2, 1, True, False)
    with pytest.raises(ValueError, match="Circularity"):
        parse_particle_settings("0-Infinity,0.5-1.1,false,false")


def test_fiji_analyze_particles_uses_fiji_output_mask(monkeypatch):
    import cellonaut.masks.roi_processing as processing

    closed = []
    calls = []
    output = SimpleNamespace(close=lambda: closed.append("output"))
    mask = SimpleNamespace(close=lambda: closed.append("input"))

    class FakeParticleAnalyzer:
        SHOW_MASKS = 4096
        EXCLUDE_EDGE_PARTICLES = 8
        INCLUDE_HOLES = 1024

        def __init__(self, *args):
            calls.append(args)

        def setHideOutputImage(self, hidden):
            assert hidden is True

        def analyze(self, image):
            assert image is mask
            return True

        def getOutputImage(self):
            return output

    monkeypatch.setattr(processing, "jimport", lambda name: FakeParticleAnalyzer if name.endswith("ParticleAnalyzer") else lambda: object())
    monkeypatch.setattr(processing, "create_mask_from_roi", lambda *_args: mask)
    monkeypatch.setattr(processing, "imageplus_to_numpy_2d", lambda _image: np.array([[255, 0], [255, 0]], dtype=np.uint8))
    monkeypatch.setattr(processing, "roi_from_mask_array", lambda array: np.asarray(array))

    result = apply_fiji_analyze_particles_step(
        "roi", object(), {"particle_settings": "20-Infinity,0.2-1,true,true"}
    )

    assert isinstance(result, np.ndarray)
    assert calls[0][0] == 4096 | 8 | 1024
    assert calls[0][3:] == (20, float("inf"), 0.2, 1)
    assert result.tolist() == [[True, False], [True, False]]
    assert closed == ["output", "input"]


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("OR", {1, 2, 3, 4}), ("AND", {2, 3}), ("XOR", {1, 4})],
)
def test_combined_mask_boolean_operations_apply_to_whole_source_masks(operation, expected):
    roi_map = {
        "a__class1": BooleanShapeRoi({1, 2}),
        "a__class2": BooleanShapeRoi({3}),
        "b": BooleanShapeRoi({2, 3, 4}),
        "c": BooleanShapeRoi({3}),
    }

    result = _combined_roi_from_sources(roi_map, ["a", "b"], operation)

    assert result is not None
    assert result.pixels == expected
    assert roi_map["b"].pixels == {2, 3, 4}
    three_source_result = _combined_roi_from_sources(roi_map, ["a", "b", "c"], "XOR")
    assert three_source_result is not None
    assert three_source_result.pixels == {1, 3, 4}
    assert _combined_roi_from_sources(roi_map, ["a", "missing"], "AND") is None
    no_overlap = _combined_roi_from_sources(
        {"a": BooleanShapeRoi({1}), "b": BooleanShapeRoi({2})}, ["a", "b"], "AND"
    )
    assert no_overlap is not None
    assert no_overlap.pixels == set()


def test_measure_roi_stats_uses_pixel_calibration_and_restores_image_metadata(monkeypatch):
    original_calibration = object()

    class FakeCalibration:
        def __init__(self):
            self.unit = ""

        def setUnit(self, unit):
            self.unit = unit

    class FakeMeasurements:
        AREA = 1
        MEAN = 2
        INTEGRATED_DENSITY = 4
        SHAPE_DESCRIPTORS = 8

    class FakeResultsTable:
        def __init__(self):
            self.values = {}

        def getValue(self, label, _row):
            if label not in self.values:
                raise KeyError(label)
            return self.values[label]

    class FakeImage:
        def __init__(self):
            self.calibration = original_calibration
            self.calibration_history = []
            self.roi_cleared = False

        def setRoi(self, _roi):
            pass

        def getCalibration(self):
            return self.calibration

        def setCalibration(self, calibration):
            self.calibration = calibration
            self.calibration_history.append(calibration)

        def killRoi(self):
            self.roi_cleared = True

    image = FakeImage()

    class FakeAnalyzer:
        def __init__(self, analyzer_image, _flags, table):
            self.image = analyzer_image
            self.table = table

        def measure(self):
            assert isinstance(self.image.calibration, FakeCalibration)
            assert self.image.calibration.unit == "pixel"
            self.table.values.update(
                {
                    "Area": 4.0,
                    "Mean": 3.0,
                    "RawIntDen": 12.0,
                    "Circ.": 0.75,
                    "Solidity": 0.9,
                }
            )

    monkeypatch.setattr(
        "cellonaut.masks.roi_processing.get_java_classes",
        lambda: {
            "Analyzer": FakeAnalyzer,
            "ResultsTable": FakeResultsTable,
            "Measurements": FakeMeasurements,
            "Calibration": FakeCalibration,
        },
    )

    stats = measure_roi_stats(image, FakeRoi())

    assert stats["Area"] == 4.0
    assert stats["Mean"] == 3.0
    assert stats["RawIntDen"] == 12.0
    assert stats["Circularity"] == 0.75
    assert stats["Solidity"] == 0.9
    assert image.calibration is original_calibration
    assert isinstance(image.calibration_history[0], FakeCalibration)
    assert image.calibration_history[-1] is original_calibration
    assert image.roi_cleared is True


def test_union_shape_rois_uses_python_bridge_or_alias():
    base = FakeShapeRoi()
    added = object()

    assert union_shape_rois(base, added) == ("union", added)
    assert base.union_arg is added


def test_union_shape_rois_raises_clear_error_without_bridge_method():
    with pytest.raises(AttributeError, match="union method is unavailable"):
        union_shape_rois(object(), object())


def test_subtract_background_closes_duplicate_when_fiji_fails(monkeypatch):
    closed = []

    class FakeDuplicate:
        def close(self):
            closed.append(True)

    class FakeDuplicator:
        def run(self, _image):
            return FakeDuplicate()

    class FakeIJ:
        @staticmethod
        def run(_image, _command, _options):
            raise RuntimeError("background failed")

    monkeypatch.setattr(
        "cellonaut.masks.roi_processing.get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )

    with pytest.raises(RuntimeError, match="background failed"):
        subtract_background_copy(object(), 8)

    assert closed == [True]


def test_roi_from_mask_array_handles_read_only_byteprocessor_pixels(monkeypatch):
    processors = []

    class FakeByteProcessor:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.pixels = np.zeros(width * height, dtype=np.int8)
            self.pixels.setflags(write=False)
            self.received_pixels = None
            self.set_calls = []
            processors.append(self)

        def getPixels(self):
            return self.pixels

        def setPixels(self, pixels):
            self.received_pixels = np.array(pixels, dtype=None, copy=True)

        def set(self, x, y, value):
            self.set_calls.append((x, y, value))

    class FakeIJ:
        @staticmethod
        def setThreshold(_imp, _low, _high):
            pass

        @staticmethod
        def run(_imp, _command, _options):
            pass

    class FakeImagePlus:
        def __init__(self, _title, processor):
            self.processor = processor

        def getRoi(self):
            return "roi"

        def close(self):
            pass

    class FakeShapeRoi:
        def __init__(self, roi):
            self.roi = roi

    monkeypatch.setattr(
        "cellonaut.masks.roi_processing.get_java_classes",
        lambda: {
            "IJ": FakeIJ,
            "ShapeRoi": FakeShapeRoi,
            "ImagePlus": FakeImagePlus,
            "ByteProcessor": FakeByteProcessor,
        },
    )

    roi = roi_from_mask_array(np.array([[1, 0], [0, 1]], dtype=np.uint8))

    assert roi is not None
    assert roi.roi == "roi"
    assert processors[0].set_calls == []
    assert np.array_equal(
        processors[0].received_pixels,
        np.array([-1, 0, 0, -1], dtype=np.int8),
    )


def test_ordered_mask_processing_recipe_applies_steps_in_saved_order(monkeypatch):
    calls = []

    def fake_binary(roi, ref_img, step_type, params):
        calls.append((step_type, roi))
        return f"{roi}|{step_type}"

    def fake_translate(roi, ref_img, params):
        calls.append(("translate", roi))
        return f"{roi}|translate"

    monkeypatch.setattr("cellonaut.masks.roi_processing.apply_fiji_binary_step", fake_binary)
    monkeypatch.setattr("cellonaut.masks.roi_processing.apply_fiji_translate_step", fake_translate)

    image_def = SimpleNamespace(
        mask_processing_steps=[
            {"type": "binary_dilate", "enabled": True, "params": {"binary_settings": "2,1,false"}},
            {"type": "translate", "enabled": True, "params": {"translate_offsets": "2,-1"}},
            {"type": "binary_erode", "enabled": True, "params": {"binary_settings": "1,1,false"}},
        ],
    )
    roi = apply_ordered_mask_processing_recipe(
        "roi",
        object(),
        image_def,
        result_id="sample",
        roi_label="Mask",
        log_func=lambda _message: None,
    )

    assert roi == "roi|binary_dilate|translate|binary_erode"
    assert [call[0] for call in calls] == ["binary_dilate", "translate", "binary_erode"]


def test_weka_threshold_uses_dedicated_mask_source_setting_not_recipe_threshold():
    image_def = SimpleNamespace(
        threshold_method="Otsu",
        mask_processing_steps=[
            {"type": "threshold_method", "enabled": True, "params": {"threshold_method": "Triangle"}},
        ],
    )
    cfg = SimpleNamespace(threshold_method="Default")

    assert mask_threshold_method_for_image(image_def, cfg) == "Otsu"
