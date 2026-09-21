from __future__ import annotations

import pytest

from cellonaut.config.defaults import IMAGE_PROCESSING_SCOPE_SEGMENTATION
from cellonaut.config.processing_steps import parse_outlier_settings
from cellonaut.pipeline.image_processing import (
    apply_imagej_processing_recipe_copy,
    imagej_processing_recipe_tiles,
    imagej_processing_recipe_signature,
)
from cellonaut.pipeline.models import ImageDef


def test_remove_outliers_uses_fiji_radius_threshold_and_polarity(monkeypatch):
    calls = []

    class FakeDuplicator:
        def run(self, image):
            return image

    class FakeIJ:
        @staticmethod
        def run(image, command, options):
            calls.append((command, options))

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "remove_outliers",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"outlier_settings": "2,50,Dark"},
            }
        ],
    )

    _result, labels = apply_imagej_processing_recipe_copy(object(), image_def, IMAGE_PROCESSING_SCOPE_SEGMENTATION)

    assert calls == [("Remove Outliers...", "radius=2 threshold=50 which=Dark")]
    assert labels == ["Remove Outliers radius=2, threshold=50, dark"]
    assert imagej_processing_recipe_signature(image_def, IMAGE_PROCESSING_SCOPE_SEGMENTATION) == "remove_outliers=2,50,Dark"


@pytest.mark.parametrize("value", ["0,50,Bright", "2,-1,Dark", "2,50,Both", "2,", "nan,5,Bright"])
def test_remove_outliers_rejects_invalid_settings(value):
    with pytest.raises(ValueError):
        parse_outlier_settings(value)


def test_apply_imagej_processing_recipe_runs_enhance_contrast_command(monkeypatch):
    calls = []
    source = object()
    duplicate = object()

    class FakeDuplicator:
        def run(self, image):
            calls.append(("duplicate", image))
            return duplicate

    class FakeIJ:
        @staticmethod
        def run(image, command, options):
            calls.append(("run", image, command, options))

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "enhance_contrast",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"contrast_saturation": "15"},
            }
        ],
    )

    result, labels = apply_imagej_processing_recipe_copy(
        source,
        image_def,
        IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    )

    assert result is duplicate
    assert labels == ["Enhance contrast saturated=15%"]
    assert calls == [
        ("duplicate", source),
        ("run", duplicate, "Enhance Contrast", "saturated=15.00"),
    ]


def test_apply_imagej_processing_recipe_runs_apply_lut_command(monkeypatch):
    calls = []
    source = object()
    duplicate = object()

    class FakeDuplicator:
        def run(self, image):
            calls.append(("duplicate", image))
            return duplicate

    class FakeIJ:
        @staticmethod
        def run(image, command, options):
            calls.append(("run", image, command, options))

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "apply_lut",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {},
            }
        ],
    )

    result, labels = apply_imagej_processing_recipe_copy(
        source,
        image_def,
        IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    )

    assert result is duplicate
    assert labels == ["Apply LUT"]
    assert calls == [
        ("duplicate", source),
        ("run", duplicate, "Apply LUT", ""),
    ]


def test_recipe_capture_preserves_each_stage_without_changing_source(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from cellonaut.pipeline import image_processing

    source = np.zeros((2, 2))
    duplicate = source.copy()
    commands = []

    def run(image, command, options):
        commands.append(command)
        image[:] += 1

    monkeypatch.setattr(image_processing, "get_java_classes", lambda: {
        "IJ": SimpleNamespace(run=run),
        "Duplicator": lambda: SimpleNamespace(run=lambda _image: duplicate),
    })
    monkeypatch.setattr(image_processing, "imageplus_to_numpy_2d", lambda image: image)
    image_def = ImageDef("signal", "Signal", "Signal", None, image_processing_steps=[
        {"type": "smooth", "enabled": True, "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
         "params": {"smooth_iterations": "1"}},
        {"type": "despeckle", "enabled": True, "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION, "params": {}},
    ])
    tiles, final, temporary = imagej_processing_recipe_tiles(source, image_def, IMAGE_PROCESSING_SCOPE_SEGMENTATION)
    assert commands == ["Smooth", "Despeckle"]
    assert len(tiles) == 2 and tiles[0][0].startswith("01 ") and tiles[1][0].startswith("02 ")
    assert final is not None
    np.testing.assert_array_equal(tiles[0][1], np.ones((2, 2)))
    np.testing.assert_array_equal(final, np.full((2, 2), 2))
    np.testing.assert_array_equal(source, np.zeros((2, 2)))
    assert len(temporary) == 1 and temporary[0] is duplicate
    assert not np.shares_memory(tiles[0][1], duplicate)


def test_empty_imagej_recipe_does_not_initialize_fiji(monkeypatch):
    def fail_if_java_is_loaded():
        raise AssertionError("Fiji must stay unused for an empty recipe")

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        fail_if_java_is_loaded,
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[],
    )
    source = object()

    result, labels = apply_imagej_processing_recipe_copy(
        source,
        image_def,
        IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    )
    tiles, final_arr, temporary_images = imagej_processing_recipe_tiles(
        source,
        image_def,
        IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    )

    assert result is source
    assert labels == []
    assert tiles == []
    assert final_arr is None
    assert temporary_images == []



def test_apply_imagej_processing_recipe_runs_common_fiji_preprocessing_commands(monkeypatch):
    calls = []
    source = object()
    duplicate = object()

    class FakeDuplicator:
        def run(self, image):
            calls.append(("duplicate", image))
            return duplicate

    class FakeIJ:
        @staticmethod
        def run(image, command, options):
            calls.append(("run", image, command, options))

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "gaussian_blur",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"gaussian_sigma": "1.5"},
            },
            {
                "type": "median",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"median_radius": "2"},
            },
            {
                "type": "despeckle",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {},
            },
            {
                "type": "bit_depth",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"bit_depth": "16-bit"},
            },
        ],
    )

    result, labels = apply_imagej_processing_recipe_copy(
        source,
        image_def,
        IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    )

    assert result is duplicate
    assert labels == [
        "Gaussian Blur sigma=1.5",
        "Median radius=2",
        "Despeckle",
        "Convert Bit Depth 16-bit",
    ]
    assert calls == [
        ("duplicate", source),
        ("run", duplicate, "Gaussian Blur...", "sigma=1.5"),
        ("run", duplicate, "Median...", "radius=2"),
        ("run", duplicate, "Despeckle", ""),
        ("run", duplicate, "16-bit", ""),
    ]


def test_apply_imagej_processing_recipe_runs_smooth_macro_command_repeatedly(monkeypatch):
    calls = []
    source = object()
    duplicate = object()

    class FakeDuplicator:
        def run(self, image):
            calls.append(("duplicate", image))
            return duplicate

    class FakeIJ:
        @staticmethod
        def run(image, command, options):
            calls.append(("run", image, command, options))

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "smooth",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"smooth_iterations": "5"},
            }
        ],
    )

    result, labels = apply_imagej_processing_recipe_copy(
        source,
        image_def,
        IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    )

    assert result is duplicate
    assert labels == ["Smooth x5"]
    assert calls == [("duplicate", source)] + [
        ("run", duplicate, "Smooth", ""),
        ("run", duplicate, "Smooth", ""),
        ("run", duplicate, "Smooth", ""),
        ("run", duplicate, "Smooth", ""),
        ("run", duplicate, "Smooth", ""),
    ]


def test_apply_imagej_processing_recipe_preserves_recipe_order(monkeypatch):
    calls = []
    source = object()
    duplicate = object()

    class FakeDuplicator:
        def run(self, image):
            calls.append(("duplicate", image))
            return duplicate

    class FakeIJ:
        @staticmethod
        def run(image, command, options):
            calls.append(("run", command, options))

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "enhance_contrast",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"contrast_saturation": "15"},
            },
            {
                "type": "smooth",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"smooth_iterations": "2"},
            },
            {
                "type": "rolling_ball_background",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"bg_radii": "5,10"},
            },
        ],
    )

    result, labels = apply_imagej_processing_recipe_copy(
        source,
        image_def,
        IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    )

    assert result is duplicate
    assert labels == [
        "Enhance contrast saturated=15%",
        "Smooth x2",
        "Subtract Background radius=5",
        "Subtract Background radius=10",
    ]
    assert calls == [
        ("duplicate", source),
        ("run", "Enhance Contrast", "saturated=15.00"),
        ("run", "Smooth", ""),
        ("run", "Smooth", ""),
        ("run", "Subtract Background...", "rolling=5.0"),
        ("run", "Subtract Background...", "rolling=10.0"),
    ]


def test_imagej_processing_recipe_signature_includes_contrast_and_rolling_ball():
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "enhance_contrast",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"contrast_saturation": "15.0"},
            },
            {
                "type": "smooth",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"smooth_iterations": "5"},
            },
            {
                "type": "rolling_ball_background",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"bg_radii": "5,10"},
            },
        ],
    )

    assert imagej_processing_recipe_signature(image_def, IMAGE_PROCESSING_SCOPE_SEGMENTATION) == (
        "enhance_contrast=15;smooth=5;subtract_background=5,10"
    )


def test_fiji_background_and_contrast_options_are_replayed_and_signed(monkeypatch):
    calls = []

    class FakeDuplicator:
        def run(self, image):
            return image

    class FakeIJ:
        @staticmethod
        def run(_image, command, options):
            calls.append((command, options))

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "enhance_contrast",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"contrast_saturation": "0.35", "normalize": True},
            },
            {
                "type": "rolling_ball_background",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {
                    "bg_radii": "40",
                    "light_background": True,
                    "sliding_paraboloid": True,
                    "disable_smoothing": True,
                },
            },
        ],
    )

    apply_imagej_processing_recipe_copy(object(), image_def, IMAGE_PROCESSING_SCOPE_SEGMENTATION)

    assert calls == [
        ("Enhance Contrast", "saturated=0.35 normalize"),
        ("Subtract Background...", "rolling=40.0 light sliding disable"),
    ]
    assert imagej_processing_recipe_signature(image_def, IMAGE_PROCESSING_SCOPE_SEGMENTATION) == (
        "enhance_contrast=0.35 normalize;subtract_background=40 light sliding disable"
    )


def test_bit_depth_conversion_restores_fiji_global_options(monkeypatch):
    observed = []

    class FakeConverter:
        scaling = True

        @classmethod
        def getDoScaling(cls):
            return cls.scaling

        @classmethod
        def setDoScaling(cls, enabled):
            cls.scaling = enabled

    class FakePrefs:
        calibrateConversions = True

    class FakeDuplicator:
        def run(self, image):
            return image

    class FakeIJ:
        @staticmethod
        def run(_image, _command, _options):
            observed.append((FakeConverter.scaling, FakePrefs.calibrateConversions))

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )
    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.jimport",
        lambda name: {"ij.process.ImageConverter": FakeConverter, "ij.Prefs": FakePrefs}[name],
    )
    image_def = ImageDef(
        key="image1", label="Signal", folder_name="Signal", model_path=None,
        image_processing_steps=[{
            "type": "bit_depth", "enabled": True, "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            "params": {"bit_depth": "8-bit", "scale_when_converting": False},
        }],
    )

    apply_imagej_processing_recipe_copy(object(), image_def, IMAGE_PROCESSING_SCOPE_SEGMENTATION)

    assert observed == [(False, False)]
    assert FakeConverter.scaling is True
    assert FakePrefs.calibrateConversions is True


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("bad", "Contrast saturation must be a number"),
        ("101", "Contrast saturation must be <= 100"),
    ],
)
def test_apply_imagej_processing_recipe_rejects_invalid_contrast_saturation(monkeypatch, value, message):
    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": object(), "Duplicator": object()},
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "enhance_contrast",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"contrast_saturation": value},
            }
        ],
    )

    with pytest.raises(ValueError, match=message):
        apply_imagej_processing_recipe_copy(
            object(),
            image_def,
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("bad", "Smooth iterations must be an integer"),
        ("0", "Smooth iterations must be >= 1"),
        ("101", "Smooth iterations must be <= 100"),
    ],
)
def test_apply_imagej_processing_recipe_rejects_invalid_smooth_iterations(monkeypatch, value, message):
    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": object(), "Duplicator": object()},
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "smooth",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"smooth_iterations": value},
            }
        ],
    )

    with pytest.raises(ValueError, match=message):
        apply_imagej_processing_recipe_copy(
            object(),
            image_def,
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        )


def test_apply_imagej_processing_recipe_closes_duplicate_when_fiji_fails(monkeypatch):
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
            raise RuntimeError("Fiji failed")

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "despeckle",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {},
            }
        ],
    )

    with pytest.raises(RuntimeError, match="Fiji failed"):
        apply_imagej_processing_recipe_copy(
            object(),
            image_def,
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        )

    assert closed == [True]


def test_imagej_processing_recipe_tiles_closes_duplicate_when_fiji_fails(monkeypatch):
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
            raise RuntimeError("Fiji failed")

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )
    image_def = ImageDef(
        key="image1",
        label="Signal",
        folder_name="Signal",
        model_path=None,
        image_processing_steps=[
            {
                "type": "despeckle",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {},
            }
        ],
    )

    with pytest.raises(RuntimeError, match="Fiji failed"):
        imagej_processing_recipe_tiles(
            object(),
            image_def,
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        )

    assert closed == [True]
