"""Contracts between run defaults, target choices, and execution snapshots."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from cellonaut.config.adapter import build_pipeline_config_from_gui_state
from cellonaut.config.defaults import INPUT_STRUCTURE_FLAT_TIFFS
from cellonaut.config.state import CellonautGuiState, ImageGuiState
from cellonaut.masks import cellpose
from cellonaut.pipeline.models import (
    Config,
    DiameterDefault,
    ImageDef,
    MeasurementTarget,
    ResolvedMeasurementTarget,
    config_for_measurement_target,
    resolve_measurement_target,
)
from cellonaut.pipeline.summary import build_pipeline_summary_dict
from cellonaut.pipeline.validation import validate_config


def make_config(tmp_path: Path, **settings) -> Config:
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    return Config(
        fiji_app_path=tmp_path / "Fiji",
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        input_structure=INPUT_STRUCTURE_FLAT_TIFFS,
        images=[ImageDef("a", "A", "A", None), ImageDef("b", "B", "B", None)],
        exclusion_tag="control",
        threshold_method="Otsu",
        probability_class_index=2,
        **settings,
    )


@pytest.mark.parametrize("blank", ["", None])
@pytest.mark.parametrize("reverse", [False, True])
def test_gui_blank_diameter_stays_original_scale_through_preset_and_segmentation(
    tmp_path: Path, monkeypatch, blank, reverse: bool,
):
    preset_images = [
        {"name": "A", "cell_diameter": "30", "analysis_cell_segmentation_enabled": True},
        {"name": "B", "cell_diameter": blank, "analysis_cell_segmentation_enabled": True},
    ]
    if reverse:
        preset_images.reverse()
    # Existing presets contain strings or null, never an inheritance sentinel.
    restored = json.loads(json.dumps(preset_images))
    restored = [ImageGuiState.from_dict(item, i).to_dict() for i, item in enumerate(restored)]
    state = CellonautGuiState.from_widget_values(
        fiji_app_path="",
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        input_structure=INPUT_STRUCTURE_FLAT_TIFFS,
        image_definitions=restored,
    )
    (tmp_path / "input").mkdir()
    cfg = build_pipeline_config_from_gui_state(state)
    validate_config(cfg)
    monkeypatch.setattr(cellpose, "cellpose_acceleration_enabled", lambda requested: requested)
    labels = {image.key: image.label for image in cfg.images}
    actual = {}
    for target in cfg.measurement_targets:
        assert isinstance(target, ResolvedMeasurementTarget)
        execution = config_for_measurement_target(cfg, target)
        actual[labels[target.source_image_key]] = cellpose.build_cell_segmentation_cfg_from_pipeline(execution).diameter
    assert actual == {"A": 30.0, "B": None}
    assert cfg.cell_diameter is None


def test_omitted_target_settings_use_explicit_run_defaults(tmp_path: Path):
    cfg = make_config(
        tmp_path,
        cell_diameter=64,
        cell_min_size=900,
        cell_use_gpu=False,
        cellprob_threshold=-0.5,
        flow_threshold=0.8,
        cell_remove_border=False,
        cellpose_model_type="cpsam_v2",
        cellpose_custom_model_path="run-model.pt",
        measurement_options={"area": False},
        cell_mask_adjustments={"dx": 3},
    )
    resolved = resolve_measurement_target(cfg, MeasurementTarget("a"))
    for name in (
        "cell_diameter", "cell_min_size", "cell_use_gpu", "cellprob_threshold", "flow_threshold",
        "cell_remove_border", "cellpose_model_type", "cellpose_custom_model_path",
        "measurement_options",
        "cell_mask_adjustments",
    ):
        assert getattr(resolved, name) == getattr(cfg, name)
    # Relationships remain target-owned choices.
    assert resolved.overlay_roi_keys == []


def test_none_empty_and_false_target_values_are_not_inheritance(tmp_path: Path):
    cfg = make_config(
        tmp_path, cell_diameter=30, cell_min_size=99, cell_use_gpu=True,
        cellprob_threshold=1, flow_threshold=1, cell_remove_border=True,
        cellpose_custom_model_path="another-channel.pt",
        measurement_options={"area": True}, cell_mask_adjustments={"dx": 5},
        overlay_roi_keys=["other"],
    )
    target = MeasurementTarget(
        "b", cell_diameter=None, cell_min_size=0, cell_use_gpu=False,
        cellprob_threshold=0, flow_threshold=0, cell_remove_border=False,
        cellpose_custom_model_path="",
        measurement_options={},
        cell_mask_adjustments={}, overlay_roi_keys=[],
    )
    effective = config_for_measurement_target(cfg, target)
    assert effective.cell_diameter is None
    assert effective.cellpose_custom_model_path == ""
    for name in ("cell_min_size", "cellprob_threshold", "flow_threshold"):
        assert getattr(effective, name) == 0
    for name in ("cell_use_gpu", "cell_remove_border"):
        assert getattr(effective, name) is False
    for name in ("measurement_options", "cell_mask_adjustments"):
        assert getattr(effective, name) == {}
    assert effective.overlay_roi_keys == []


def test_target_and_execution_collections_are_independently_owned(tmp_path: Path):
    cfg = make_config(tmp_path, cell_diameter=30, measurement_options={"area": True},
                      cell_mask_adjustments={"dx": 3})
    a = resolve_measurement_target(cfg, MeasurementTarget("a", cell_min_size=20))
    b = resolve_measurement_target(cfg, MeasurementTarget("b", cell_min_size=40, cell_diameter=None))
    a.cell_mask_adjustments["dx"] = 10
    a.measurement_options["area"] = False
    assert b.cell_mask_adjustments == cfg.cell_mask_adjustments == {"dx": 3}
    assert b.measurement_options == cfg.measurement_options == {"area": True}
    effective = config_for_measurement_target(cfg, b)
    effective.cell_mask_adjustments["dx"] = 100
    assert b.cell_mask_adjustments == cfg.cell_mask_adjustments
    assert (a.cell_min_size, b.cell_min_size) == (20, 40)
    assert (a.cell_diameter, b.cell_diameter) == (30, None)


def test_validation_resolves_once_before_execution_and_preserves_run_resources(tmp_path: Path):
    target = MeasurementTarget("a", do_cell_segmentation=True)
    cfg = make_config(tmp_path, cell_diameter=30, cell_min_size=20, measurement_targets=[target])
    validate_config(cfg)
    resolved = cfg.measurement_targets[0]
    assert isinstance(resolved, ResolvedMeasurementTarget)
    assert resolved.cell_segmentation_source == "a"
    cfg.cell_diameter = 90
    cfg.cell_min_size = 80
    assert resolve_measurement_target(cfg, resolved) is resolved
    effective = config_for_measurement_target(cfg, resolved)
    assert effective.cell_diameter == 30
    assert effective.cell_min_size == 20
    for name in ("input_dir", "output_dir", "fiji_app_path", "images", "input_structure", "exclusion_tag", "threshold_method", "probability_class_index"):
        assert getattr(effective, name) == getattr(cfg, name)
    # Omitted values on the unresolved input were not silently rewritten.
    assert target.cell_diameter is DiameterDefault.INHERIT


def test_validation_checks_inherited_values_but_not_an_explicit_unset_diameter(tmp_path: Path):
    cfg = make_config(tmp_path, cell_diameter=-10, measurement_targets=[MeasurementTarget("a", do_cell_segmentation=True)])
    with pytest.raises(ValueError, match="Cell diameter"):
        validate_config(cfg)
    cfg.measurement_targets = [MeasurementTarget("a", do_cell_segmentation=True, cell_diameter=None)]
    validate_config(cfg)
    assert config_for_measurement_target(cfg, cfg.measurement_targets[0]).cell_diameter is None


def test_summary_and_spawn_serialization_preserve_explicit_configuration(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("cellonaut.pipeline.summary.build_runtime_metadata", lambda _cfg: {})
    cfg = make_config(tmp_path, cell_diameter=30, measurement_options={"area": True})
    cfg.measurement_targets = [MeasurementTarget("a"), MeasurementTarget("b", cell_diameter=None, measurement_options={})]
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored.measurement_targets[0].cell_diameter is DiameterDefault.INHERIT
    summary = json.loads(json.dumps(build_pipeline_summary_dict(restored)))
    assert [target["cell_diameter"] for target in summary["measurement_targets"]] == [30, None]
    assert summary["measurement_targets"][1]["measurement_options"] == {}
    assert summary["configuration_snapshot"]["measurement_targets"][0]["cell_diameter"] == "use_run_default"
    validate_config(restored)
    restored = pickle.loads(pickle.dumps(restored))
    assert all(isinstance(target, ResolvedMeasurementTarget) for target in restored.measurement_targets)
    assert config_for_measurement_target(restored, restored.measurement_targets[1]).cell_diameter is None
