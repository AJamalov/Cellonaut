from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cellonaut.exceptions import PipelineCancelled
from cellonaut.measurement import execution
from cellonaut.pipeline.models import ImageDef


def test_sample_overlay_export_always_writes_all_outputs(monkeypatch):
    calls = []
    monkeypatch.setattr(execution, "save_general_overlay_stack", lambda **kwargs: calls.append(("tiff", kwargs)))
    monkeypatch.setattr(execution, "save_general_flat_qc_overlay", lambda **kwargs: calls.append(("png", kwargs)))

    execution.save_sample_overlay_outputs(
        cfg=SimpleNamespace(
            overlay_roi_keys=["mask"],
            images=[ImageDef("mask", "Mask", "Mask", None)],
            do_cell_segmentation=False,
            overlay_whole_cell_mask=False,
        ),
        image_map={},
        roi_map={"mask": object()},
        extra_overlay_masks={},
        export_dirs={
            "mask_overlays": Path("overlays"),
            "final_binary_masks": Path("masks"),
            "qc_overlay_pngs": Path("previews"),
        },
        result_id="sample",
        id_label="sample",
        overlay_base_img=object(),
        overlay_base_def=SimpleNamespace(label="Signal"),
        native_numpy_images=False,
        warning_text="",
        log_func=lambda _message: None,

    )

    assert len(calls) == 2
    assert calls[0][0] == "tiff"
    assert calls[0][1]["write_overlay"] is True
    assert calls[1][0] == "png"


def test_required_binary_mask_export_failure_propagates(monkeypatch):
    monkeypatch.setattr(
        execution,
        "save_general_overlay_stack",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        execution.save_sample_overlay_outputs(
            cfg=SimpleNamespace(
                overlay_roi_keys=["mask"],
            images=[ImageDef("mask", "Mask", "Mask", None)],
                do_cell_segmentation=False,
                overlay_whole_cell_mask=False,
            ),
            image_map={},
            roi_map={"mask": object()},
            extra_overlay_masks={},
            export_dirs={
                "mask_overlays": Path("overlays"),
                "final_binary_masks": Path("masks"),
                "qc_overlay_pngs": Path("previews"),
            },
            result_id="sample",
            id_label="sample",
            overlay_base_img=object(),
            overlay_base_def=SimpleNamespace(label="Signal"),
            native_numpy_images=False,
            warning_text="",
            log_func=lambda _message: None,

        )


def test_background_subtracted_measurement_is_auto_written(monkeypatch):
    def fake_measure_roi_stats(image, _roi):
        if image == "corrected":
            return {"Area": 4.0, "Mean": 2.0, "RawIntDen": 40.0}
        return {"Area": 4.0, "Mean": 3.0, "RawIntDen": 100.0}

    monkeypatch.setattr(execution, "measure_roi_stats", fake_measure_roi_stats)

    row = {}
    source_def = SimpleNamespace(key="signal", label="Signal")
    roi_def = SimpleNamespace(key="mask", label="Mask")

    execution.measure_rois_for_source(
        cfg=SimpleNamespace(measurement_options={"raw_intden": False}),
        roi_defs=[roi_def],
        roi_map={"mask": object()},
        row=row,
        source_def=source_def,
        source_measure_img="raw",
        measurement_source_img="corrected",
        has_source_background_subtraction=True,
        corrected_suffix="RB40p0",
        id_label="sample",
        log_func=lambda _message: None,

    )

    assert row["Signal_in_Mask_RB40p0_IntDen"] == 40.0
    assert "Signal_in_Mask_RawIntDen" not in row
    assert row["Signal_in_Mask_Area"] == 4.0


def test_configured_mask_circularity_and_solidity_are_selectable(monkeypatch):
    monkeypatch.setattr(
        execution,
        "measure_roi_stats",
        lambda *_args: {"Circularity": 0.75, "Solidity": 0.9},
    )
    row = {}

    execution.measure_rois_for_source(
        cfg=SimpleNamespace(
            measurement_options={"circularity": True, "solidity": True}
        ),
        roi_defs=[SimpleNamespace(key="mask", label="Mask")],
        roi_map={"mask": object()},
        row=row,
        source_def=SimpleNamespace(key="signal", label="Signal"),
        source_measure_img="raw",
        measurement_source_img="raw",
        has_source_background_subtraction=False,
        corrected_suffix="RB40p0",
        id_label="sample",
        log_func=lambda _message: None,
    )

    assert row["Signal_in_Mask_Circularity"] == 0.75
    assert row["Signal_in_Mask_Solidity"] == 0.9


@pytest.mark.parametrize("outcome", ["success", "measurement_error", "cancel"])
def test_background_sweep_disposes_owned_images(imagej_measurement_backend, outcome):
    backend = imagej_measurement_backend
    source = backend.Image([[1, 2], [3, 4]])
    primary_image = backend.Image([[5, 6], [7, 8]])
    temporary_images = [primary_image]  # Owned by the caller, not this sweep.

    def fail_measurement(_image):
        raise RuntimeError("measurement failed")

    if outcome == "measurement_error":
        backend.on_measure = fail_measurement
    row = {}

    def run():
        execution.run_source_background_sweep(
            roi_defs=[SimpleNamespace(key="mask", label="Mask")], roi_map={"mask": backend.roi},
            row=row, source_def=SimpleNamespace(label="Signal"), measurement_base_img=source,
            source_radii=[40.0, 70.0, 90.0], source_options=["", " light", " disable"],
            created_temp_images=temporary_images, id_label="sample", log_func=lambda _message: None,
            # Cancellation between radii must not strand the previous image or
            # create the next one. Measurement errors exercise the inner finally.
            should_cancel=lambda: outcome == "cancel" and bool(backend.measured),
        )

    if outcome == "measurement_error":
        with pytest.raises(RuntimeError, match="measurement failed"):
            run()
    elif outcome == "cancel":
        with pytest.raises(PipelineCancelled):
            run()
    else:
        run()

    expected_options = ["rolling=70.0 light"]
    if outcome == "success":
        expected_options.append("rolling=90.0 disable")
    assert [(command, options) for _image, command, options in backend.commands] == [
        ("Subtract Background...", options) for options in expected_options
    ]
    assert len(backend.copies) == len(expected_options)
    assert all(parent is source and image.closes == 1 for parent, image in backend.copies)
    assert source.closes == primary_image.closes == 0
    assert temporary_images == [primary_image]
    expected_row = {} if outcome == "measurement_error" else {"Signal_in_Mask_RB70p0_IntDen": 10.0}
    if outcome == "success":
        expected_row["Signal_in_Mask_RB90p0_IntDen"] = 10.0
    assert row == expected_row
