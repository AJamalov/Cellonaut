from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from cellonaut.gui.preview_artifacts import CellonautGuiPreviewArtifactsMixin
from cellonaut.masks import cellpose, preview_filter_overlay as filters
from cellonaut.masks.overlay_exports import save_general_overlay_stack
from cellonaut.pipeline.output_manifest import save_run_manifest
from cellonaut.pipeline.preview_artifacts import collect_preview_artifacts
from cellonaut.results.artifacts import ArtifactResolver, ArtifactMetadataError, record_artifact
from cellonaut.results.layout import build_results_layout


@pytest.fixture
def written_results(tmp_path, monkeypatch):
    labels = np.array([[1, 1, 0], [0, 2, 2], [0, 2, 2]], dtype=np.int32)
    monkeypatch.setattr(cellpose, "run_cell_segmentation_on_image", lambda *_args, **_kwargs: labels)
    source = SimpleNamespace(key="image1", label="Channel_with_underscores")
    cfg = SimpleNamespace(
        do_cell_segmentation=True,
        cell_segmentation_source="image1",
        source_image_key="image1",
        cell_diameter=None,
        cell_min_size=1,
        cell_use_gpu=False,
        cellpose_model_type="cpsam",
        cellpose_custom_model_path="",
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        cell_remove_border=False,
        measurement_options={"cell_area": True},
        reuse_existing_masks=False,
        cell_mask_adjustments={},
        images=[source],
    )
    layout = build_results_layout(tmp_path)
    dirs = {key: value / "nested" / "sample_folder" for key, value in layout.items()}
    source_image = np.arange(9, dtype=np.uint16).reshape(3, 3)
    row = {}
    masks, _warning = cellpose.run_cell_segmentation_for_sample(
        cfg,
        {"image1": source_image},
        {},
        [],
        row,
        source,
        source_image,
        source_image,
        False,
        dirs,
        "sample_with_underscores",
        "sample",
        lambda _message: None,
    )
    save_general_overlay_stack(
        base_img=source_image,
        roi_map={},
        roi_keys=["__whole_cell_mask__"],
        out_path=dirs["mask_overlays"],
        binary_out_path=dirs["final_binary_masks"],
        result_id="sample_with_underscores",
        base_label=source.label,
        cfg=cfg,
        log_func=lambda _message: None,
        extra_mask_map=masks,
        image_map={"image1": source_image},
    )
    return layout["root"]


def no_legacy(*_args, **_kwargs):
    raise AssertionError("Metadata-backed results must not guess filenames")


def test_writers_record_exact_relationships_and_summary_preserves_them(written_results):
    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    overlay = resolver.matching(kind="combined_overlay")[0]
    assert overlay["target"] == "image1"
    assert overlay["sample"] == "sample_with_underscores"
    assert "nested/sample_folder/" in overlay["path"]
    assert resolver.path(overlay).name == "sample_with_underscores_Channel_with_underscores_combined_overlay.tif"
    assert resolver.sidecar(overlay)["layer_keys"][0] == "image1"
    for kind in ("cell_table", "cell_labels", "cell_outline", "cell_png"):
        assert resolver.path(resolver.related(overlay, kind)).is_file()
    save_run_manifest(
        written_results, run_type="preview", pipeline_summary={"runtime_environment": {"python": {"version": "3"}}}
    )
    updated = ArtifactResolver.load(written_results)
    assert updated is not None and updated.records == resolver.records
    assert not any(Path(item["path"]).is_absolute() for item in resolver.records)


@pytest.mark.parametrize("moved_folder", ["Results", "Renamed_results"])
def test_preview_navigation_export_and_reuse_share_metadata_after_move_and_rename(
    written_results, tmp_path, monkeypatch, moved_folder
):
    moved = tmp_path / "moved" / moved_folder
    moved.parent.mkdir()
    shutil.move(str(written_results), moved)
    resolver = ArtifactResolver.load(moved)
    assert resolver is not None
    overlay = resolver.path(resolver.matching(kind="combined_overlay")[0])
    definitions = [
        {
            "key": "image1",
            "name": "Renamed_display",
            "cell_populations": [
                {"name": "Large", "cell_qc_limits": "Area:3-", "exclude_from_csv": True},
            ],
        }
    ]
    from cellonaut.config.adapter import parse_cell_qc_limits_text as parse_cell_qc_rules

    for helper in (
        "_find_legacy_preview_filter_data",
        "_split_table_base",
        "_find_signal_tables",
        "_best_scored_artifact",
        "_table_base_from_path",
    ):
        monkeypatch.setattr(filters, helper, no_legacy)
    before = resolver.path(resolver.related(resolver.record(overlay), "cell_table")).read_bytes()
    data, message = filters.find_preview_filter_data(overlay, definitions)
    assert message == ""
    assert data is not None and data.source_label == "Renamed_display"
    exported = filters.export_all_filtered_result_tables(moved, definitions, parse_cell_qc_rules)
    assert exported.processed_count == 1, exported.messages
    assert exported.kept_combined_path is not None
    table = pd.read_csv(exported.kept_combined_path)
    assert table["CellID"].tolist() == [1]
    assert data.table_path.read_bytes() == before
    loaded = cellpose.load_existing_cellpose_labels(moved, data.result_id, "Renamed_display", source_key="image1")
    assert loaded is not None
    assert np.array_equal(loaded, data.label_image)
    harness = SimpleNamespace(
        preview_artifact_results_roots=lambda: [moved],
        preview_artifact_mode_paths=no_legacy,
        preview_artifact_sample_key=no_legacy,
    )
    entries = CellonautGuiPreviewArtifactsMixin.collect_preview_artifact_entries(
        cast(CellonautGuiPreviewArtifactsMixin, harness), "overlay"
    )
    assert entries[0]["sample"] == "sample_with_underscores"
    assert entries[0]["path"] == str(overlay)
    assert collect_preview_artifacts(moved)["preview_path"] == str(overlay)


def test_metadata_pair_is_authoritative_even_with_unrelated_exact_looking_files(written_results, monkeypatch):
    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    overlay = resolver.matching(kind="combined_overlay")[0]
    labels = resolver.related(overlay, "cell_labels")
    label_path = resolver.path(labels)
    # A valid-looking other target must never substitute for the recorded mask.
    label_path.rename(label_path.with_name("other_Channel_with_underscores_01_cellpose_labels.tif"))
    monkeypatch.setattr(filters, "_find_legacy_preview_filter_data", no_legacy)
    data, message = filters.find_preview_filter_data(resolver.path(overlay), [])
    assert data is None
    assert "missing" in message
    export = filters.export_all_filtered_result_tables(written_results, [], lambda _text: {})
    assert export.processed_count == 0 and export.skipped_count == 1
    assert "missing" in export.messages[0]


@pytest.mark.parametrize(
    "section",
    [
        None,
        {},
        {"version": 2, "files": []},
        {"version": 1, "files": [{}]},
        {"version": 1, "files": [{"path": "../outside", "sample": "s", "target": "t", "kind": "cell_labels"}]},
    ],
)
def test_invalid_metadata_does_not_fall_back(written_results, monkeypatch, section):
    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    overlay = resolver.path(resolver.matching(kind="combined_overlay")[0])
    manifest = written_results / "Logs" / "RunSummary.json"
    manifest.write_text(json.dumps({"artifacts": section}), encoding="utf-8")
    monkeypatch.setattr(filters, "_find_legacy_preview_filter_data", no_legacy)
    data, message = filters.find_preview_filter_data(overlay, [])
    assert data is None and "Could not resolve saved artifacts" in message


def test_missing_sidecar_is_reported_without_filename_fallback(written_results):
    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    overlay = resolver.matching(kind="combined_overlay")[0]
    resolver.path(overlay, "sidecar").unlink()
    with pytest.raises(ArtifactMetadataError, match="missing"):
        filters._load_preview_sidecar(resolver.path(overlay))
    # Table/label relationships remain usable without presentation metadata.
    data, message = filters.find_preview_filter_data(resolver.path(overlay), [])
    assert data is not None and not message


def test_legacy_loading_is_read_only_and_retains_exact_pair_protection(written_results):
    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    overlay = resolver.path(resolver.matching(kind="combined_overlay")[0])
    manifest = written_results / "Logs" / "RunSummary.json"
    manifest.write_text('{"app_version": "old"}', encoding="utf-8")
    before = manifest.read_bytes()
    data, message = filters.find_preview_filter_data(overlay, [{"name": "Channel_with_underscores"}])
    assert data is not None, message
    assert manifest.read_bytes() == before
    data.labels_path.unlink()
    assert filters.find_preview_filter_data(overlay, [{"name": "Channel_with_underscores"}])[0] is None


def test_writer_rejects_conflicting_identity(written_results):
    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    record = resolver.matching(kind="cell_table")[0]
    with pytest.raises(ArtifactMetadataError, match="Conflicting"):
        record_artifact(resolver.path(record), sample="another_sample", target=record["target"], kind="cell_table")


@pytest.mark.parametrize("source_mode, expected", [("Measured image", [1]), ("Cell mask image", [2])])
def test_preview_and_batch_mask_filtering_use_saved_keys_after_display_rename(
    written_results, source_mode, expected, monkeypatch
):
    import tifffile
    from cellonaut.config.adapter import parse_cell_qc_limits_text
    from cellonaut.gui.preview_filter import CellonautGuiPreviewFilterMixin
    from cellonaut.gui.state import PreviewState
    from cellonaut.masks.cell_groups import evaluate_cell_groups

    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    record = resolver.matching(kind="combined_overlay")[0]
    overlay = resolver.path(record)
    # Saved measurements differ from raw overlay pixels. Measured-source groups
    # must use those measurements; cell-source groups must use the selected plane.
    stack = np.array(
        [
            [[5, 5, 0], [0, 5, 5], [0, 5, 5]],
            [[100, 100, 0], [0, 10, 10], [0, 10, 10]],
            [[1, 1, 0], [0, 1, 1], [0, 1, 1]],
        ],
        dtype=np.uint16,
    )
    tifffile.imwrite(overlay, stack, metadata={"axes": "CYX"}, photometric="minisblack")
    sidecar = {
        "layer_labels": ["Old_measured", "Old_source", "Old_mask"],
        "layer_keys": ["image1", "image2", "image3"],
        "layer_roles": ["image", "image", "weka_mask"],
    }
    resolver.path(record, "sidecar").write_text(json.dumps(sidecar), encoding="utf-8")
    signal_path = written_results / "arbitrary" / "table_without_channel_or_sample.csv"
    signal_path.parent.mkdir()
    pd.DataFrame({"CellID": [1, 2], "Old_maskMeanGrayValue_InCell": [10, 100]}).to_csv(signal_path, index=False)
    record_artifact(
        signal_path,
        sample=record["sample"],
        target="image1",
        label="Old_measured",
        kind="cell_signal",
        mask="image3",
        mask_label="Old_mask",
    )
    settings = {
        "key": "image1",
        "name": "New_measured",
        "analysis_cell_segmentation_source": "New_source",
        "mask_relationships": {"New_mask": True},
        "cell_group_mask_source": "New_mask",
        "cell_populations": [
            {
                "name": "Bright",
                "mask_qc_limits": "Mean intensity inside mask:50-",
                "mask_qc_intensity_source": source_mode,
                "exclude_from_csv": True,
            }
        ],
    }
    definitions = [settings, {"key": "image2", "name": "New_source"}, {"key": "image3", "name": "New_mask"}]
    monkeypatch.setattr(filters, "_find_legacy_preview_filter_data", no_legacy)
    monkeypatch.setattr(filters, "_split_table_base", no_legacy)
    monkeypatch.setattr(filters, "_find_signal_tables", no_legacy)
    data, message = filters.find_preview_filter_data(overlay, definitions)
    assert data is not None, message
    assert data.table["MeanInPositiveArea"].iloc[:2].tolist() == [10, 100]
    state = PreviewState()
    state.file_path = str(overlay)
    state.current_labels = sidecar["layer_labels"]
    state.current_layer_keys = sidecar["layer_keys"]
    state.current_layer_roles = sidecar["layer_roles"]
    state.tiff_model = {"data": stack[None, None]}
    harness = SimpleNamespace(
        preview_state=state,
        parse_cell_qc_limits_text=parse_cell_qc_limits_text,
        get_active_image_definitions=lambda: definitions,
        _preview_filter_label_key=CellonautGuiPreviewFilterMixin._preview_filter_label_key,
    )
    preview = evaluate_cell_groups(
        data.table,
        settings,
        parse_cell_qc_limits_text,
        prepare_table=lambda table, group: CellonautGuiPreviewFilterMixin.attach_preview_mask_filter_metrics(
            cast(CellonautGuiPreviewFilterMixin, harness), table, data.label_image, group
        ),
    )
    preview.require_available()
    assert sorted(preview.kept_labels) == expected
    result = filters.export_all_filtered_result_tables(written_results, definitions, parse_cell_qc_limits_text)
    assert result.processed_count == 1, result.messages
    assert result.kept_combined_path is not None
    assert pd.read_csv(result.kept_combined_path)["CellID"].tolist() == expected
    assert result.signal_combined_path is not None
    assert pd.read_csv(result.signal_combined_path)["CellID"].tolist() == expected


@pytest.mark.parametrize("text", ['{"artifacts":', "[]", '{"artifacts":{"version":1,"files":[]}}'])
def test_corrupt_or_partial_manifest_cannot_reconstruct_unrecorded_files(written_results, text, monkeypatch):
    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    overlay = resolver.path(resolver.matching(kind="combined_overlay")[0])
    (written_results / "Logs" / "RunSummary.json").write_text(text, encoding="utf-8")
    monkeypatch.setattr(filters, "_find_legacy_preview_filter_data", no_legacy)
    data, message = filters.find_preview_filter_data(overlay, [])
    assert data is None and message


def test_artifacts_can_be_relocated_inside_results_without_filename_inference(written_results, monkeypatch):
    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    for index, record in enumerate(resolver.records):
        path = resolver.path(record)
        new_path = written_results / "relocated" / str(index) / f"opaque{path.suffix}"
        new_path.parent.mkdir(parents=True)
        path.rename(new_path)
        record["path"] = new_path.relative_to(written_results).as_posix()
    manifest = written_results / "Logs" / "RunSummary.json"
    manifest.write_text(json.dumps({"artifacts": {"version": 1, "files": resolver.records}}), encoding="utf-8")
    overlay = resolver.path(resolver.matching(kind="combined_overlay")[0])
    monkeypatch.setattr(filters, "_find_legacy_preview_filter_data", no_legacy)
    monkeypatch.setattr(filters, "_table_base_from_path", no_legacy)
    data, message = filters.find_preview_filter_data(overlay, [{"key": "image1", "name": "Renamed"}])
    assert data is not None, message
    result = filters.export_all_filtered_result_tables(
        written_results, [{"key": "image1", "name": "Renamed"}], lambda _text: {}
    )
    assert result.processed_count == 1, result.messages


def test_new_empty_execution_does_not_use_legacy_discovery(tmp_path, monkeypatch):
    from cellonaut.results.artifacts import initialize_artifact_manifest
    from cellonaut.pipeline import preview_artifacts

    root = tmp_path / "Results"
    initialize_artifact_manifest(root)
    monkeypatch.setattr(preview_artifacts, "_collect_legacy_preview_artifacts", no_legacy)
    assert preview_artifacts.collect_preview_artifacts(root)["preview_path"] is None


def test_missing_signal_skips_sample_in_batch_and_reports_failure(written_results):
    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    record = resolver.matching(kind="cell_table")[0]
    path = written_results / "signal.csv"
    path.write_text("CellID,MaskArea_InCell\n1,2\n", encoding="utf-8")
    record_artifact(
        path, sample=record["sample"], target=record["target"], kind="cell_signal", mask="image2", mask_label="Mask"
    )
    path.unlink()
    result = filters.export_all_filtered_result_tables(
        written_results, [{"key": "image1", "name": "New"}], lambda _text: {}
    )
    assert result.processed_count == 0 and result.skipped_count == 1
    assert "missing" in result.messages[0]


def test_recorded_weka_masks_keep_classifier_and_class_checks(tmp_path):
    import tifffile
    from cellonaut.masks.roi_processing import threshold_mask_path
    from cellonaut.pipeline.processing_montage import _artifact_arrays

    root = build_results_layout(tmp_path)["root"]
    path = root / "arbitrary.tif"
    tifffile.imwrite(path, np.ones((4, 5), dtype=np.uint8))
    record_artifact(
        path, sample="sample", target="image3", label="Original_mask", kind="threshold", mask="2", classifier="model"
    )
    assert (
        threshold_mask_path(root / "Masks" / "MaskImages", "sample", "Renamed_mask", "model", 2, target_key="image3")
        == path
    )
    with pytest.raises(ArtifactMetadataError, match="No unique"):
        threshold_mask_path(root, "sample", "Renamed_mask", "other_model", 2, target_key="image3")
    with pytest.raises(ArtifactMetadataError, match="No unique"):
        threshold_mask_path(root, "sample", "Renamed_mask", "model", 1, target_key="image3")
    tiles = _artifact_arrays(
        probability_dir=None,
        threshold_dir=root,
        result_id="sample",
        label="Renamed_mask",
        target_key="image3",
        class_index=2,
    )
    assert len(tiles) == 1 and tiles[0][0] == "Threshold mask"


def test_metadata_publish_failure_keeps_previous_inventory(written_results, monkeypatch):
    import cellonaut.io.writers as writers

    manifest = written_results / "Logs" / "RunSummary.json"
    before = manifest.read_bytes()
    path = written_results / "pending.tif"
    path.write_bytes(b"published media")
    monkeypatch.setattr(
        writers, "_publish_temporary_output", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("publish failed"))
    )
    with pytest.raises(OSError, match="publish failed"):
        record_artifact(path, sample="sample", target="image1", kind="cell_labels")
    assert manifest.read_bytes() == before
    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    with pytest.raises(ArtifactMetadataError, match="not recorded"):
        resolver.record(path)
    assert not list(manifest.parent.glob("*.tmp*"))


def test_corrupt_sidecar_and_unregistered_user_snapshot_are_distinct(written_results):
    from cellonaut.results.artifacts import load_artifact_sidecar

    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    overlay = resolver.matching(kind="combined_overlay")[0]
    resolver.path(overlay, "sidecar").write_text("{", encoding="utf-8")
    with pytest.raises(ArtifactMetadataError, match="sidecar"):
        load_artifact_sidecar(resolver.path(overlay))
    snapshot = written_results / "Image Preview Tools" / "Snapshots" / "my_snapshot.tif"
    snapshot.parent.mkdir(parents=True)
    snapshot.touch()
    assert load_artifact_sidecar(snapshot) == {}
    assert filters.find_preview_filter_data(snapshot, [])[0] is None


@pytest.mark.parametrize("valid_thin_image", [False, True])
def test_preview_and_export_agree_on_label_image_loading(written_results, valid_thin_image):
    import tifffile

    resolver = ArtifactResolver.load(written_results)
    assert resolver is not None
    record = resolver.matching(kind="combined_overlay")[0]
    labels = resolver.path(resolver.related(record, "cell_labels"))
    if valid_thin_image:
        tifffile.imwrite(labels, np.array([[1, 1, 2, 2]], dtype=np.int32))
    else:
        labels.write_bytes(b"corrupt label TIFF")
    definitions = [{"key": "image1", "name": "Current"}]
    data, message = filters.find_preview_filter_data(resolver.path(record), definitions)
    result = filters.export_all_filtered_result_tables(written_results, definitions, lambda _text: {})
    if valid_thin_image:
        assert data is not None and data.label_image.shape == (1, 4), message
        assert result.processed_count == 1, result.messages
    else:
        assert data is None and message
        assert result.processed_count == 0 and result.skipped_count == 1
        assert result.messages
