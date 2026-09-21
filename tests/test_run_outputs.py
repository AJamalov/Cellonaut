import json
from pathlib import Path

import pytest

from cellonaut.pipeline.models import Config
from cellonaut.pipeline.run_outputs import config_for_numbered_child_output, next_numbered_child


def test_next_numbered_child_uses_first_available_number(tmp_path: Path):
    (tmp_path / "preview_1").mkdir()
    (tmp_path / "preview_2").mkdir()

    assert next_numbered_child(tmp_path, "preview") == tmp_path / "preview_3"


def test_next_numbered_child_starts_at_one(tmp_path: Path):
    assert next_numbered_child(tmp_path, "run") == tmp_path / "run_1"


def test_numbered_child_preserves_mask_source_for_reuse_runs(tmp_path: Path):
    cfg = Config(
        fiji_app_path=tmp_path / "Fiji",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        input_structure="Samples directly in input folder",
        images=[],
        exclusion_tag="_ut_",
        threshold_method="Default",
        probability_class_index=1,
        reuse_existing_masks=True,
    )

    rerun_cfg = config_for_numbered_child_output(cfg, "run")

    assert rerun_cfg.output_dir == tmp_path / "output" / "run_1"
    assert rerun_cfg.mask_source_dir == tmp_path / "output"
    state_path = rerun_cfg.output_dir / "Results" / "Logs" / "RunState.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["operation_type"] == "full"
    assert state["state"] == "in_progress"


def test_numbered_child_removes_empty_reservation_when_state_write_fails(tmp_path: Path, monkeypatch):
    cfg = Config(
        fiji_app_path=tmp_path / "Fiji",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        input_structure="Samples directly in input folder",
        images=[],
        exclusion_tag="",
        threshold_method="Default",
        probability_class_index=1,
    )

    def fail_state_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("cellonaut.pipeline.run_outputs.save_operation_state", fail_state_write)

    with pytest.raises(OSError, match="disk full"):
        config_for_numbered_child_output(cfg, "run")

    assert not (cfg.output_dir / "run_1").exists()
