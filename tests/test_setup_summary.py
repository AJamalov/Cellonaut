from types import SimpleNamespace

import pytest

from cellonaut.gui import validation
from cellonaut.gui.validation import SetupCheckWorker, CellonautGuiValidationMixin, complete_configuration_summary
from cellonaut.runtime import PipelineStage


def test_format_configuration_summary_text_uses_structured_summary():
    summary = {
        "setup_checks": [
            {
                "status": "OK",
                "check": "Input folder",
                "detail": "2 supported image files found.",
            }
        ],
        "folders": {
            "input": "C:/images",
            "output": "C:/results",
            "fiji": "C:/Fiji",
            "layout": "flat_tiff",
        },
        "images": [
            {
                "type": "Channel",
                "name": "GFP",
                "folder": "GFP",
                "classifier": "gfp.classifier",
                "probability_class": "1",
            }
        ],
        "relationships": [
            {"source": "GFP", "target": "Mask", "self": False},
        ],
        "measurements": [
            {"source": "All measured channels", "enabled": ["area", "mean"]},
        ],
        "cell_segmentation": [{
            "source": "GFP", "seg_source": "Brightfield", "diameter": "30", "min_size": "50",
            "cellprob": "0.0", "flow": "0.4", "remove_border": False,
        }],
        "filters": [{
            "source": "GFP group", "cell_limits": "Area:50-", "mask_limits": "",
            "mode": "Match all", "mask_intensity_source": "Measured channel", "exclude": True,
        }],
        "dry_run": {
            "available": True,
            "total": 2,
            "ready": 2,
            "blocked": 0,
        },
        "warnings": [],
    }

    text = CellonautGuiValidationMixin.format_configuration_summary_text(summary)

    assert "Input folder: C:/images" in text
    assert "GFP measured with Mask mask" in text
    assert "All measured channels: area, mean" in text
    assert "Detected: 2 | Ready: 2 | Blocked: 0" in text
    assert "[OK] Input folder: 2 supported image files found." in text
    assert "GFP_Cellpose: source=Brightfield, diameter=30" in text
    assert "remove border=No" in text
    assert "GFP group: cell conditions=Area:50-" in text
    assert "exclude from CSV=Yes" in text


def test_setup_check_worker_completes_filesystem_checks_from_snapshot(monkeypatch):
    cfg = SimpleNamespace(input_dir="C:/images", input_structure="flat", output_dir="C:/results")
    dry_run = {"available": True, "total": 2, "ready": 2, "blocked": 0}
    monkeypatch.setattr(validation, "analyze_sample_readiness", lambda received: dry_run if received is cfg else {})
    monkeypatch.setattr(
        validation,
        "create_setup_check_items",
        lambda *_args, **_kwargs: [{"status": "OK", "check": "Samples", "detail": "2 ready"}],
    )
    worker = SetupCheckWorker(
        {"folders": {}, "setup_checks": [], "dry_run": {}},
        cfg,
        [],
        [],
        [],
        cached_fiji_scan={},
        cached_input_scan={},
        nd2_detected_channel_names=[],
    )
    emitted = []
    worker.done_signal.connect(emitted.append)

    worker.run()

    assert emitted[0]["cancelled"] is False
    assert emitted[0]["error"] == ""
    assert emitted[0]["summary"]["dry_run"] == dry_run
    assert emitted[0]["summary"]["setup_checks"][0]["check"] == "Samples"
    assert "2 ready" in emitted[0]["text"]


def test_setup_check_worker_honors_cancellation_before_scanning(monkeypatch):
    monkeypatch.setattr(
        validation,
        "analyze_sample_readiness",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("cancelled worker must not scan")),
    )
    worker = SetupCheckWorker(
        {},
        SimpleNamespace(input_dir="C:/images", input_structure="flat"),
        [],
        [],
        [],
        cached_fiji_scan={},
        cached_input_scan={},
        nd2_detected_channel_names=[],
    )
    emitted = []
    worker.done_signal.connect(emitted.append)

    worker.request_cancel()
    worker.run()

    assert emitted == [{"cancelled": True, "error": ""}]


def test_setup_check_cleanup_defers_result_until_thread_references_are_cleared():
    emitted = []

    class SignalStub:
        def emit(self, result):
            emitted.append(result)

    class Harness(CellonautGuiValidationMixin):
        _pending_setup_check_result = {"cancelled": False, "summary": {}}
        _setup_check_worker = object()
        _setup_check_thread = object()
        _close_retry_scheduled = False
        _worker_status_base = ""
        setup_check_result_on_gui = SignalStub()

        def set_setup_check_busy(self, busy):
            pass

    harness = Harness()
    harness.cleanup_setup_check()

    assert harness._setup_check_worker is None
    assert harness._setup_check_thread is None
    assert harness._pending_setup_check_result is None
    assert emitted == [{"cancelled": False, "summary": {}}]


@pytest.mark.parametrize("wording", ["Checking setup", "Validating configuration"])
@pytest.mark.parametrize("stage", [PipelineStage.CHECKING.value, PipelineStage.SCANNING.value,
                                   PipelineStage.INITIALIZING.value, None])
def test_setup_cleanup_uses_stage_identity_and_preserves_other_status(wording, stage):
    statuses = []
    busy_states = []
    class Harness(CellonautGuiValidationMixin):
        _worker_status_stage = stage
        _worker_status_base = wording
        _setup_check_worker = object()
        _setup_check_thread = object()

        def set_setup_check_busy(self, busy):
            busy_states.append(busy)

        def update_worker_status(self, status):
            statuses.append(status)

    harness = Harness()
    harness.cleanup_setup_check()

    assert statuses == (["Ready"] if stage == PipelineStage.CHECKING else [])
    assert busy_states == [False]
    assert harness._setup_check_worker is None
    assert harness._setup_check_thread is None
    if stage != PipelineStage.CHECKING:
        assert harness._worker_status_stage == stage
        assert harness._worker_status_base == wording


def test_setup_check_cleanup_does_not_open_report_during_window_shutdown():
    emitted = []

    class SignalStub:
        def emit(self, result):
            emitted.append(result)

    class Harness(CellonautGuiValidationMixin):
        _pending_setup_check_result = {"cancelled": False, "summary": {}}
        _setup_check_worker = object()
        _setup_check_thread = object()
        _close_retry_scheduled = True
        _worker_status_base = ""
        setup_check_result_on_gui = SignalStub()

        def set_setup_check_busy(self, busy):
            pass

    harness = Harness()
    harness.cleanup_setup_check()

    assert harness._setup_check_thread is None
    assert emitted == []


def test_complete_configuration_summary_keeps_report_available_when_discovery_fails(monkeypatch, tmp_path):
    cfg = SimpleNamespace(input_dir=tmp_path, input_structure="flat")
    monkeypatch.setattr(
        validation,
        "analyze_sample_readiness",
        lambda _cfg: (_ for _ in ()).throw(OSError("folder unavailable")),
    )
    captured = {}

    def checks(_cfg, active_defs, relationships, warnings, dry_run, **kwargs):
        captured.update(
            active_defs=active_defs,
            relationships=relationships,
            warnings=warnings,
            dry_run=dry_run,
            kwargs=kwargs,
        )
        return [{"status": "BLOCKED", "check": "Samples", "detail": dry_run["error"]}]

    monkeypatch.setattr(validation, "create_setup_check_items", checks)
    summary = complete_configuration_summary(
        {"folders": {"input": str(tmp_path)}},
        cfg,
        [{"name": "GFP"}],
        [{"source": "GFP", "target": "GFP"}],
        ["warning"],
        cached_fiji_scan={"status": "cached"},
        cached_input_scan={"files": 2},
        nd2_detected_channel_names=["GFP"],
    )

    assert summary["dry_run"]["available"] is False
    assert summary["dry_run"]["error"] == "folder unavailable"
    assert summary["setup_checks"][0]["detail"] == "folder unavailable"
    assert captured["kwargs"]["nd2_detected_channel_names"] == ["GFP"]


def test_configuration_summary_snapshot_captures_relationships_cellpose_and_filters(tmp_path):
    class SummaryHarness(CellonautGuiValidationMixin):
        measurement_options = {"area": True, "mean": False}

        def collect_config(self):
            return SimpleNamespace(
                input_dir=tmp_path,
                output_dir=tmp_path / "output",
                fiji_app_path=tmp_path / "Fiji.app",
                input_structure="flat",
            )

        def get_active_image_definitions(self):
            return [
                {
                    "name": "GFP",
                    "folder": "GFP",
                    "mask_relationships": {"Mask": True},
                    "analysis_cell_segmentation_enabled": True,
                    "analysis_cell_segmentation_source": "GFP",
                        "cell_populations": [
                            {
                                "name": "Cell group 1",
                                "cell_qc_limits": "Area:1:20",
                                "exclude_from_csv": True,
                            }
                        ],
                },
                {
                    "name": "Mask",
                    "folder": "Mask",
                    "is_mask_only": True,
                    "classifier": "C:/models/mask.model",
                    "probability_class_index": "2",
                },
            ]

        def get_analysis_matrix_warnings(self):
            return ["review filters"]

        @staticmethod
        def is_physical_channel_definition(image_def):
            return not image_def.get("is_mask_only", False)

    _cfg, _defs, relationships, warnings, summary = SummaryHarness().build_configuration_summary_snapshot()

    assert relationships == [{"source": "GFP", "target": "Mask", "self": False}]
    assert warnings == ["review filters"]
    assert summary["measurements"] == [{"source": "All measured channels", "enabled": ["area"]}]
    assert summary["images"][1]["classifier_name"] == "mask.model"
    assert summary["cell_segmentation"][0]["source"] == "GFP"
    assert summary["filters"][0]["exclude"] is True


def test_pipeline_readiness_blocks_empty_and_all_incomplete_runs(monkeypatch):
    gui = CellonautGuiValidationMixin()
    gui.validate_config_before_output = lambda cfg: True
    critical = []
    monkeypatch.setattr(validation.QMessageBox, "critical", lambda *args: critical.append(args))

    monkeypatch.setattr(validation, "analyze_sample_readiness", lambda _cfg: {"total": 0, "ready": 0})
    assert gui.validate_pipeline_readiness_for_config(object()) is False
    assert critical[-1][1] == "No samples found"

    monkeypatch.setattr(
        validation,
        "analyze_sample_readiness",
        lambda _cfg: {
            "total": 2,
            "ready": 0,
            "blocked": 2,
            "problems": [{"missing_required": ["GFP/sample.tif"]}],
        },
    )
    assert gui.validate_pipeline_readiness_for_config(object()) is False
    assert critical[-1][1] == "No runnable samples"
    assert "GFP/sample.tif" in critical[-1][2]


def test_pipeline_readiness_asks_before_running_partial_batch(monkeypatch):
    gui = CellonautGuiValidationMixin()
    gui.validate_config_before_output = lambda cfg: True
    prompts = []
    monkeypatch.setattr(
        validation,
        "analyze_sample_readiness",
        lambda _cfg: {
            "total": 3,
            "ready": 2,
            "blocked": 1,
            "problems": [{"sample": "sample-3", "messages": ["missing mask"]}],
        },
    )
    monkeypatch.setattr(
        validation.QMessageBox,
        "question",
        lambda *args: prompts.append(args) or validation.QMessageBox.StandardButton.Yes,
    )

    assert gui.validate_pipeline_readiness_for_config(object()) is True
    assert prompts[0][1] == "Some samples are blocked"
    assert "sample-3: missing mask" in prompts[0][2]
