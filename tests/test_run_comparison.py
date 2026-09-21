from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cellonaut.results.comparison import compare_runs, discover_comparable_runs, resolve_comparable_run


def write_run(
    root: Path,
    number: int,
    rows: list[dict],
    *,
    input_dir: str = "C:/data",
    config: dict | None = None,
    fingerprint: str = "abc",
) -> Path:
    run = root / f"run_{number}"
    csv_dir = run / "Results" / "CSV Data"
    log_dir = run / "Results" / "Logs"
    csv_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(csv_dir / "Measurements.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "app_version": "1.0.0",
        "input_directory": input_dir,
        "configuration_snapshot": {"input_dir": input_dir, "output_dir": str(run), **(config or {})},
        "measurement_targets": [{"source_image_key": "image1", "cell_diameter": 40}],
        "runtime_environment": {
            "dependency_versions": {"numpy": "2.0"},
            "configured_file_fingerprints": [
                {"role": "classifier:Mask", "path": "classifier.model", "exists": True, "sha256": fingerprint}
            ],
        },
    }
    (log_dir / "RunSummary.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_compare_runs_aligns_samples_and_preserves_missing_values(tmp_path: Path):
    baseline = write_run(
        tmp_path,
        1,
        [
            {"Sample": "C:/data/sample_a.tif", "GFP_in_Mask_MeanGrayValue": 10, "Count": 0},
            {"Sample": "C:/data/sample_b.tif", "GFP_in_Mask_MeanGrayValue": 20, "Count": 2},
        ],
    )
    current = write_run(
        tmp_path,
        2,
        [
            {"Sample": "D:/moved/sample_a.tif", "GFP_in_Mask_MeanGrayValue": 15, "NewMetric": 4},
            {"Sample": "D:/moved/sample_b.tif", "GFP_in_Mask_MeanGrayValue": 10, "NewMetric": 5},
        ],
        input_dir="D:/moved",
    )

    result = compare_runs(baseline, current)
    changes = {(row.sample, row.measurement_key): row for row in result.measurement_changes}

    increase = changes[("sample_a.tif", "GFP_in_Mask_MeanGrayValue")]
    assert increase.difference == 5
    assert increase.percent_change == 50
    assert increase.status == "Changed"

    decrease = changes[("sample_b.tif", "GFP_in_Mask_MeanGrayValue")]
    assert decrease.difference == -10
    assert decrease.percent_change == -50

    removed = changes[("sample_a.tif", "Count")]
    assert removed.baseline == 0
    assert removed.current is None
    assert removed.status == "Removed"

    added = changes[("sample_a.tif", "NewMetric")]
    assert added.baseline is None
    assert added.current == 4
    assert added.status == "Added"


def test_compare_runs_keeps_zero_baseline_percentage_undefined(tmp_path: Path):
    baseline = write_run(tmp_path, 1, [{"Sample": "C:/data/a.tif", "Value": 0}])
    current = write_run(tmp_path, 2, [{"Sample": "C:/data/a.tif", "Value": 5}])

    change = compare_runs(baseline, current).measurement_changes[0]

    assert change.difference == 5
    assert change.percent_change is None


def test_compare_runs_reports_effective_settings_and_fingerprint_changes(tmp_path: Path):
    baseline = write_run(tmp_path, 1, [{"Sample": "C:/data/a.tif", "Value": 1}], fingerprint="old")
    current = write_run(
        tmp_path,
        2,
        [{"Sample": "C:/data/a.tif", "Value": 1}],
        config={"exclusion_tag": "_control_"},
        fingerprint="new",
    )

    result = compare_runs(baseline, current)
    settings = {row.setting: row for row in result.setting_changes}

    assert settings["Configuration > exclusion_tag"].status == "Added"
    assert settings["Configured files > classifier:Mask > sha256"].baseline == "old"
    assert settings["Configured files > classifier:Mask > sha256"].current == "new"
    assert not any("output_dir" in key for key in settings)


def test_discover_comparable_runs_includes_numbered_runs_and_previews(tmp_path: Path):
    run_10 = write_run(tmp_path, 10, [{"Sample": "C:/data/a.tif", "Value": 1}])
    run_2 = write_run(tmp_path, 2, [{"Sample": "C:/data/a.tif", "Value": 1}])
    preview = write_run(tmp_path, 3, [{"Sample": "C:/data/a.tif", "Value": 1}])
    preview.rename(tmp_path / "preview_3")

    runs = discover_comparable_runs(run_10 / "Results" / "CSV Data")

    assert [run.operation_dir for run in runs] == [run_2, run_10, tmp_path / "preview_3"]
    assert [run.operation_type for run in runs] == ["run", "run", "preview"]
    assert resolve_comparable_run(run_2 / "Results" / "Logs" / "RunSummary.json").operation_dir == run_2


def test_compare_runs_accepts_completed_previews(tmp_path: Path):
    preview_1 = write_run(tmp_path, 1, [{"Sample": "C:/data/a.tif", "Value": 1}])
    preview_1.rename(tmp_path / "preview_1")
    preview_2 = write_run(tmp_path, 2, [{"Sample": "C:/data/a.tif", "Value": 3}])
    preview_2.rename(tmp_path / "preview_2")

    result = compare_runs(tmp_path / "preview_1", tmp_path / "preview_2")

    assert result.baseline_run.operation_type == "preview"
    assert result.current_run.operation_type == "preview"
    assert result.measurement_changes[0].difference == 2


def test_compare_runs_rejects_conflicting_duplicate_measurements(tmp_path: Path):
    baseline = write_run(
        tmp_path,
        1,
        [
            {"Sample": "C:/data/a.tif", "Value": 1},
            {"Sample": "C:/data/a.tif", "Value": 2},
        ],
    )
    current = write_run(tmp_path, 2, [{"Sample": "C:/data/a.tif", "Value": 1}])

    with pytest.raises(ValueError, match="conflicting duplicate"):
        compare_runs(baseline, current)
