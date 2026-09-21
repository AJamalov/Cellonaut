from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from cellonaut.config.defaults import INPUT_STRUCTURE_FLAT_TIFFS
from cellonaut.exceptions import PipelineCancelled
from cellonaut.masks import roi_processing
from cellonaut.masks.cellpose import load_existing_cellpose_labels
from cellonaut.pipeline.models import Config, ImageDef, MeasurementTarget
from cellonaut.pipeline.readiness import analyze_sample_readiness


def test_load_existing_cellpose_labels_reads_saved_tiff(tmp_path: Path):
    labels = np.array([[0, 1], [2, 2]], dtype=np.uint16)
    label_path = tmp_path / "Results" / "Cells" / "TIFF Labels" / "SampleA_Signal_01_cellpose_labels.tif"
    label_path.parent.mkdir(parents=True)
    tifffile.imwrite(label_path, labels)

    loaded = load_existing_cellpose_labels(tmp_path, "SampleA", "Signal")

    assert loaded is not None
    assert loaded.dtype == np.int32
    assert np.array_equal(loaded, labels)


def test_load_existing_cellpose_labels_uses_matching_nested_results(tmp_path: Path):
    labels = np.array([[0, 4], [4, 0]], dtype=np.uint16)
    label_path = (
        tmp_path / "Results" / "Cells" / "TIFF Labels" / "condition_a" / "SampleA_Signal_01_cellpose_labels.tif"
    )
    label_path.parent.mkdir(parents=True)
    tifffile.imwrite(label_path, labels)

    loaded = load_existing_cellpose_labels(
        tmp_path,
        "SampleA",
        "Signal",
        relative_subdir=Path("condition_a"),
    )

    assert loaded is not None
    assert np.array_equal(loaded, labels)


def test_load_existing_weka_roi_opens_threshold_mask(monkeypatch, tmp_path: Path):
    mask_path = tmp_path / "SampleA_Mask_class1_thr_classifier.tif"
    mask_path.write_bytes(b"mask")
    opened = []

    class FakeProcessor:
        def getHistogram(self):
            return [1, 0, 1]

    class FakeImage:
        def getProcessor(self):
            return FakeProcessor()

        def getRoi(self):
            return "roi"

        def close(self):
            opened.append("closed")

    class FakeIJ:
        @staticmethod
        def openImage(path):
            opened.append(path)
            return FakeImage()

        @staticmethod
        def setThreshold(_image, _minimum, _maximum):
            return None

        @staticmethod
        def run(_image, _command, _options):
            return None

    monkeypatch.setattr(
        roi_processing,
        "get_java_classes",
        lambda: {"IJ": FakeIJ, "ShapeRoi": lambda roi: ("shape", roi)},
    )

    rois = roi_processing.load_existing_class_rois(
        tmp_path,
        "SampleA",
        "Mask",
        "classifier",
        1,
        lambda _message: None,
    )

    assert rois == {1: ("shape", "roi")}
    assert str(mask_path) in opened
    assert "closed" in opened


def test_load_existing_weka_roi_rejects_non_binary_threshold_mask(monkeypatch, tmp_path: Path):
    mask_path = tmp_path / "SampleA_Mask_class1_thr_classifier.tif"
    mask_path.write_bytes(b"mask")
    messages = []
    closed = []

    class FakeProcessor:
        def getHistogram(self):
            return [1, 1, 0]

    class FakeImage:
        def getProcessor(self):
            return FakeProcessor()

        def getRoi(self):
            raise AssertionError("Non-binary masks should not be converted to ROIs")

        def close(self):
            closed.append(True)

    class FakeIJ:
        @staticmethod
        def openImage(_path):
            return FakeImage()

    monkeypatch.setattr(
        roi_processing,
        "get_java_classes",
        lambda: {"IJ": FakeIJ, "ShapeRoi": lambda roi: ("shape", roi)},
    )

    rois = roi_processing.load_existing_class_rois(
        tmp_path,
        "SampleA",
        "Mask",
        "classifier",
        1,
        messages.append,
    )

    assert rois == {}
    assert closed == [True]
    assert any("not binary" in message for message in messages)


def test_load_existing_weka_roi_rejects_mask_when_binary_check_fails(monkeypatch, tmp_path: Path):
    mask_path = tmp_path / "SampleA_Mask_class1_thr_classifier.tif"
    mask_path.write_bytes(b"mask")
    messages = []
    closed = []

    class FakeProcessor:
        def getHistogram(self):
            raise RuntimeError("histogram unavailable")

    class FakeImage:
        def getProcessor(self):
            return FakeProcessor()

        def getRoi(self):
            raise AssertionError("Unverified masks should not be converted to ROIs")

        def close(self):
            closed.append(True)

    class FakeIJ:
        @staticmethod
        def openImage(_path):
            return FakeImage()

    monkeypatch.setattr(
        roi_processing,
        "get_java_classes",
        lambda: {"IJ": FakeIJ, "ShapeRoi": lambda roi: ("shape", roi)},
    )

    rois = roi_processing.load_existing_class_rois(
        tmp_path,
        "SampleA",
        "Mask",
        "classifier",
        1,
        messages.append,
    )

    assert rois == {}
    assert closed == [True]
    assert any("Could not verify" in message for message in messages)


def test_load_existing_weka_roi_rejects_mask_when_threshold_setup_fails(monkeypatch, tmp_path: Path):
    mask_path = tmp_path / "SampleA_Mask_class1_thr_classifier.tif"
    mask_path.write_bytes(b"mask")
    messages = []
    closed = []

    class FakeProcessor:
        def getHistogram(self):
            return [1, 0, 1]

    class FakeImage:
        def getProcessor(self):
            return FakeProcessor()

        def getRoi(self):
            raise AssertionError("Masks without a threshold should not be converted to ROIs")

        def close(self):
            closed.append(True)

    class FakeIJ:
        @staticmethod
        def openImage(_path):
            return FakeImage()

        @staticmethod
        def setThreshold(_image, _minimum, _maximum):
            raise RuntimeError("threshold unavailable")

    monkeypatch.setattr(
        roi_processing,
        "get_java_classes",
        lambda: {"IJ": FakeIJ, "ShapeRoi": lambda roi: ("shape", roi)},
    )

    rois = roi_processing.load_existing_class_rois(
        tmp_path,
        "SampleA",
        "Mask",
        "classifier",
        1,
        messages.append,
    )

    assert rois == {}
    assert closed == [True]
    assert any("Could not prepare" in message for message in messages)


def test_prepare_rois_reuses_matching_nested_mask_folder(monkeypatch, tmp_path: Path):
    output_dir = tmp_path / "run_2"
    source_dir = tmp_path / "run_1"
    out_path = output_dir / "Results" / "Masks" / "condition_a"
    captured = {}

    image_def = SimpleNamespace(
        key="image2",
        label="Mask",
        model_path=tmp_path / "classifier.model",
        probability_class_index=1,
        threshold_method="Default",
        bg_radii_csv="",
    )
    cfg = SimpleNamespace(
        output_dir=output_dir,
        mask_source_dir=source_dir,
        reuse_existing_masks=True,
        probability_class_index=1,
        threshold_method="Default",
        images=[image_def],
    )

    def fake_load(threshold_dir, *_args, **_kwargs):
        captured["threshold_dir"] = threshold_dir
        return {1: "roi"}

    monkeypatch.setattr(roi_processing, "load_existing_class_rois", fake_load)

    roi_map, _measure_images = roi_processing.prepare_rois_for_defs(
        image_map={},
        cfg=cfg,
        out_path=out_path,
        result_id="SampleA",
        roi_defs=[image_def],
        segs={},
        log_func=lambda _message: None,
    )

    assert roi_map == {"image2": "roi"}
    assert captured["threshold_dir"] == (source_dir / "Results" / "Masks" / "condition_a" / "MaskImages")


def test_prepare_rois_combines_source_masks_with_or(monkeypatch, tmp_path: Path):
    class FakeRoi:
        def __init__(self, values):
            self.values = set(values)

        def clone(self):
            return FakeRoi(self.values)

    monkeypatch.setattr(
        roi_processing,
        "union_shape_rois",
        lambda base, added: FakeRoi(base.values | added.values),
    )
    processed_labels = []

    def record_mask_processing(roi, _img, image_def, **_kwargs):
        processed_labels.append(image_def.label)
        return roi

    monkeypatch.setattr(roi_processing, "apply_ordered_mask_processing_recipe", record_mask_processing)
    first = SimpleNamespace(
        key="image1",
        label="Green",
        model_path=tmp_path / "green.model",
        probability_class_index=1,
        threshold_method="Default",
        bg_radii_csv="",
    )
    second = SimpleNamespace(
        key="image2",
        label="Red",
        model_path=tmp_path / "red.model",
        probability_class_index=1,
        threshold_method="Default",
        bg_radii_csv="",
    )
    combined = SimpleNamespace(
        key="image3",
        label="Either",
        model_path=None,
        combined_mask_source_keys=["image1", "image2"],
    )
    cfg = SimpleNamespace(
        output_dir=tmp_path,
        mask_source_dir=None,
        reuse_existing_masks=True,
        probability_class_index=1,
        threshold_method="Default",
        images=[first, second, combined],
    )
    monkeypatch.setattr(
        roi_processing,
        "load_existing_class_rois",
        lambda _path, _result, tag, *_args, **_kwargs: {1: FakeRoi({tag})},
    )

    roi_map, _measure_images = roi_processing.prepare_rois_for_defs(
        image_map={},
        cfg=cfg,
        out_path=tmp_path / "Masks",
        result_id="SampleA",
        roi_defs=[first, second, combined],
        segs={},
        log_func=lambda _message: None,
    )

    assert roi_map["image3"].values == {"Green", "Red"}
    assert processed_labels == ["Green", "Red", "Either"]


def test_prepare_rois_closes_weka_result_when_cancelled_after_classifier(monkeypatch, tmp_path: Path):
    classifier_path = tmp_path / "classifier.model"
    classifier_path.write_bytes(b"model")
    closed = []

    class FakeClassifierResult:
        def close(self):
            closed.append(True)

    class FakeSegmenter:
        def applyClassifier(self, _image, _threads, _probability):
            return FakeClassifierResult()

    image_def = SimpleNamespace(
        key="image2",
        label="Mask",
        model_path=classifier_path,
        probability_class_index=1,
        threshold_method="Default",
        combined_mask_source_keys=[],
        mask_processing_steps=[],
        image_processing_steps=[],
    )
    cfg = SimpleNamespace(
        output_dir=tmp_path / "output",
        mask_source_dir=None,
        reuse_existing_masks=False,
        probability_class_index=1,
        threshold_method="Default",
        images=[image_def],
    )
    cancel_checks = iter([False, True])
    monkeypatch.setattr(
        roi_processing,
        "apply_imagej_processing_recipe_copy",
        lambda *_args, **_kwargs: (object(), []),
    )

    with pytest.raises(PipelineCancelled):
        roi_processing.prepare_rois_for_defs(
            image_map={"image2": object()},
            cfg=cfg,
            out_path=tmp_path / "Masks",
            result_id="SampleA",
            roi_defs=[image_def],
            segs={"image2": FakeSegmenter()},
            log_func=lambda _message: None,
            file_map={"image2": tmp_path / "sample.tif"},
            should_cancel=lambda: next(cancel_checks),
        )

    assert closed == [True]


@pytest.mark.parametrize("empty", [False, True])
def test_build_roi_saves_binary_mask_instead_of_probability_image(
    monkeypatch,
    tmp_path: Path,
    empty: bool,
):
    saved_objects = []
    closed_objects = []

    class FakeProcessor:
        def duplicate(self):
            return self

        def setAutoThreshold(self, _method):
            return None

    class FakeStack:
        def getProcessor(self, _index):
            return FakeProcessor()

    class FakeResult:
        def getStackSize(self):
            return 1

        def getStack(self):
            return FakeStack()

    class FakeProbabilityImage:
        def __init__(self, _title, _processor):
            self.roi = None if empty else "threshold-roi"

        def getProcessor(self):
            return FakeProcessor()

        def getRoi(self):
            return self.roi

        def getWidth(self):
            return 4

        def getHeight(self):
            return 3

        def close(self):
            closed_objects.append("probability")

    class FakeSavedMask:
        def close(self):
            closed_objects.append("mask")

    class FakeFileSaver:
        def __init__(self, image):
            saved_objects.append(image)

        def saveAsTiff(self, path):
            Path(path).write_bytes(b"mask")
            return True

    class FakeIJ:
        @staticmethod
        def run(_image, _command, _options):
            return None

    binary_mask = FakeSavedMask()
    monkeypatch.setattr(
        roi_processing,
        "get_java_classes",
        lambda: {
            "IJ": FakeIJ,
            "ImagePlus": FakeProbabilityImage,
            "FileSaver": FakeFileSaver,
            "ShapeRoi": lambda roi: ("shape", roi),
        },
    )
    monkeypatch.setattr(
        roi_processing,
        "create_mask_from_roi",
        lambda _image, roi, _title: binary_mask if roi == (None if empty else ("shape", "threshold-roi")) else None,
    )

    output = tmp_path / "Results" / "Masks" / "MaskImages"
    rois = roi_processing.build_roi(
        FakeResult(),
        output,
        "SampleA",
        "Mask",
        "classifier",
        "Default",
        1,
        lambda _message: None,
        target_key="image2",
    )

    assert rois == {1: None if empty else ("shape", "threshold-roi")}
    assert saved_objects == [binary_mask]
    assert closed_objects == ["mask", "probability"]
    assert (output / "SampleA_Mask_class1_thr_classifier.tif").read_bytes() == b"mask"

    from cellonaut.results.artifacts import ArtifactResolver

    resolver = ArtifactResolver.load(output)
    assert resolver is not None
    record = resolver.matching(sample="SampleA", target="image2", kind="threshold")[0]
    assert record["mask"] == "1" and record["classifier"] == "classifier"
    assert resolver.path(record) == output / "SampleA_Mask_class1_thr_classifier.tif"


def test_build_roi_closes_probability_and_mask_when_export_fails(monkeypatch, tmp_path: Path):
    closed_objects = []

    class FakeProcessor:
        def duplicate(self):
            return object()

        def setAutoThreshold(self, _method):
            pass

    class FakeStack:
        def getProcessor(self, _index):
            return FakeProcessor()

    class FakeResult:
        def getStackSize(self):
            return 1

        def getStack(self):
            return FakeStack()

    class FakeProbabilityImage:
        def __init__(self, _title, _processor):
            self.processor = FakeProcessor()

        def getProcessor(self):
            return self.processor

        def getRoi(self):
            return "threshold-roi"

        def close(self):
            closed_objects.append("probability")

    class FakeSavedMask:
        def close(self):
            closed_objects.append("mask")

    class FakeIJ:
        @staticmethod
        def run(_image, _command, _options):
            pass

    saved_mask = FakeSavedMask()
    monkeypatch.setattr(
        roi_processing,
        "get_java_classes",
        lambda: {
            "IJ": FakeIJ,
            "ImagePlus": FakeProbabilityImage,
            "FileSaver": object(),
            "ShapeRoi": lambda roi: ("shape", roi),
        },
    )
    monkeypatch.setattr(roi_processing, "create_mask_from_roi", lambda *_args: saved_mask)
    monkeypatch.setattr(
        roi_processing,
        "save_imagej_tiff",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("export failed")),
    )

    with pytest.raises(OSError, match="export failed"):
        roi_processing.build_roi(
            FakeResult(),
            tmp_path,
            "SampleA",
            "Mask",
            "classifier",
            "Default",
            1,
            lambda _message: None,
        )

    assert closed_objects == ["mask", "probability"]


def test_build_roi_rejects_unavailable_weka_class_before_export(monkeypatch, tmp_path: Path):
    class FakeResult:
        @staticmethod
        def getStackSize():
            return 2

    monkeypatch.setattr(
        roi_processing,
        "get_java_classes",
        lambda: {"IJ": object(), "ImagePlus": object(), "FileSaver": object(), "ShapeRoi": object()},
    )

    with pytest.raises(ValueError, match=r"class number\(s\) 3.*choose 1-2"):
        roi_processing.build_roi(
            FakeResult(),
            tmp_path,
            "SampleA",
            "Mask",
            "classifier",
            "Default",
            "1,3",
            lambda _message: None,
        )


def test_prepare_rois_caches_skeleton_metrics_for_final_mask(
    monkeypatch,
    tmp_path: Path,
):
    closed_masks = []
    image_def = SimpleNamespace(
        key="image2",
        label="Tubules",
        model_path=tmp_path / "classifier.model",
        probability_class_index=1,
        threshold_method="Default",
        bg_radii_csv="",
        mask_processing_steps=[{"type": "analyze_skeleton", "enabled": True, "params": {}}],
    )
    cfg = SimpleNamespace(
        output_dir=tmp_path / "output",
        mask_source_dir=tmp_path / "source",
        reuse_existing_masks=True,
        probability_class_index=1,
        threshold_method="Default",
        images=[image_def],
    )
    metrics = {}
    skeleton_paths = []

    class FakeMask:
        def close(self):
            closed_masks.append(True)

    monkeypatch.setattr(
        roi_processing,
        "load_existing_class_rois",
        lambda *_args, **_kwargs: {1: "final-roi"},
    )
    monkeypatch.setattr(
        roi_processing,
        "create_mask_from_roi",
        lambda *_args, **_kwargs: FakeMask(),
    )

    def fake_analyze_skeleton(_mask, skeleton_path=None):
        skeleton_paths.append(skeleton_path)
        return {
            "SkeletonCount": 2,
            "BranchCount": 8,
            "EndpointCount": 4,
            "JunctionCount": 3,
        }

    monkeypatch.setattr(
        roi_processing,
        "analyze_mask_skeleton",
        fake_analyze_skeleton,
    )

    roi_map, _measure_images = roi_processing.prepare_rois_for_defs(
        image_map={"image2": object()},
        cfg=cfg,
        out_path=tmp_path / "output" / "Results" / "Masks",
        result_id="SampleA",
        roi_defs=[image_def],
        segs={},
        log_func=lambda _message: None,
        skeleton_metrics=metrics,
    )

    assert roi_map == {"image2": "final-roi"}
    assert metrics == {
        "Tubules_MaskSkeleton_SkeletonCount": 2,
        "Tubules_MaskSkeleton_BranchCount": 8,
        "Tubules_MaskSkeleton_EndpointCount": 4,
        "Tubules_MaskSkeleton_JunctionCount": 3,
    }
    assert skeleton_paths == [tmp_path / "output" / "Results" / "Masks" / "Skeletons" / "SampleA_Tubules_skeleton.tif"]
    assert closed_masks == [True]


def test_readiness_warns_when_reusable_mask_name_does_not_match(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "SampleA.tif").write_bytes(b"image")
    classifier = tmp_path / "classifier.model"
    classifier.write_text("model", encoding="utf-8")
    cfg = Config(
        fiji_app_path=tmp_path / "Fiji",
        input_dir=input_dir,
        output_dir=tmp_path / "run_2",
        input_structure=INPUT_STRUCTURE_FLAT_TIFFS,
        images=[
            ImageDef("image1", "Signal", "Signal", None),
            ImageDef("image2", "Mask", "Mask", classifier),
        ],
        exclusion_tag="_ut_",
        threshold_method="Default",
        probability_class_index=1,
        measurement_targets=[
            MeasurementTarget(
                source_image_key="image1",
                overlay_roi_keys=["image2"],
            )
        ],
        reuse_existing_masks=True,
        mask_source_dir=tmp_path / "run_1",
    )

    readiness = analyze_sample_readiness(cfg)

    assert readiness["warning_count"] == 1
    warnings = readiness["samples"][0]["reuse_mask_warnings"]
    assert len(warnings) == 1
    assert "SampleA_Mask_class1_thr_classifier.tif" in warnings[0]
    assert "channel label" in warnings[0]
