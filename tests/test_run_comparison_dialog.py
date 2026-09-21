from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog, QTableWidget, QTableWidgetItem  # noqa: E402

from cellonaut.gui.run_comparison import RunComparisonDialog  # noqa: E402
from cellonaut.gui import run_comparison as run_comparison_module  # noqa: E402


pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


def write_run(root: Path, number: int, rows: list[dict]) -> Path:
    run = root / f"run_{number}"
    csv_dir = run / "Results" / "CSV Data"
    log_dir = run / "Results" / "Logs"
    csv_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(csv_dir / "Measurements.csv", index=False)
    manifest = {
        "app_version": "1.0.0",
        "input_directory": "C:/data",
        "configuration_snapshot": {"input_dir": "C:/data", "output_dir": str(run)},
        "measurement_targets": [],
        "runtime_environment": {},
    }
    (log_dir / "RunSummary.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


# Assert once at the Qt boundary so individual expectations stay focused on color behavior.
def table_item(table: QTableWidget, row: int, column: int) -> QTableWidgetItem:
    item = table.item(row, column)
    assert item is not None
    return item


# Exercise the rendered table because the requested colors and default filter
# are user-visible behavior that the data-only comparison tests cannot verify.
def test_dialog_uses_directional_colors_and_filters_unchanged_rows(qt_app: QApplication, tmp_path: Path):
    write_run(
        tmp_path,
        1,
        [
            {"Sample": "C:/data/a.tif", "Value": 10},
            {"Sample": "C:/data/b.tif", "Value": 20},
            {"Sample": "C:/data/c.tif", "Value": 4},
        ],
    )
    write_run(
        tmp_path,
        2,
        [
            {"Sample": "C:/data/a.tif", "Value": 15},
            {"Sample": "C:/data/b.tif", "Value": 10},
            {"Sample": "C:/data/c.tif", "Value": 4},
        ],
    )

    dialog = RunComparisonDialog(start_path=tmp_path)

    assert dialog.comparison is not None
    assert dialog.measurement_table.rowCount() == 2
    rows = {
        table_item(dialog.measurement_table, row, 0).text(): row for row in range(dialog.measurement_table.rowCount())
    }
    increase_row = rows["a.tif"]
    decrease_row = rows["b.tif"]
    assert table_item(dialog.measurement_table, increase_row, 2).foreground().color().name() == "#f3f3f3"
    assert table_item(dialog.measurement_table, increase_row, 3).foreground().color().name() == "#4caf50"
    assert table_item(dialog.measurement_table, decrease_row, 3).foreground().color().name() == "#d96c6c"

    dialog.changed_only_checkbox.setChecked(False)
    assert dialog.measurement_table.rowCount() == 3


# Export the filtered rows rather than all rows so a saved comparison remains
# consistent with the view the user prepared in the dialog.
def test_dialog_exports_visible_measurement_rows(
    qt_app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_run(tmp_path, 1, [{"Sample": "C:/data/a.tif", "Value": 10}, {"Sample": "C:/data/b.tif", "Value": 20}])
    write_run(tmp_path, 2, [{"Sample": "C:/data/a.tif", "Value": 15}, {"Sample": "C:/data/b.tif", "Value": 10}])
    dialog = RunComparisonDialog(start_path=tmp_path)
    dialog.measurement_filter.setText("a.tif")
    output = tmp_path / "comparison.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: (str(output), "CSV files (*.csv)"))

    dialog.export_measurement_comparison()

    exported = pd.read_csv(output)
    assert exported["Sample"].tolist() == ["a.tif"]
    assert exported["Difference"].tolist() == [5.0]


# Opening comparison from a preview should keep preview tuning iterations
# together even when completed full runs exist beside them.
def test_dialog_prefers_preview_pair_when_opened_from_preview(qt_app: QApplication, tmp_path: Path):
    write_run(tmp_path, 1, [{"Sample": "C:/data/a.tif", "Value": 1}])
    write_run(tmp_path, 2, [{"Sample": "C:/data/a.tif", "Value": 2}])
    preview_3 = write_run(tmp_path, 3, [{"Sample": "C:/data/a.tif", "Value": 3}])
    preview_3.rename(tmp_path / "preview_3")
    preview_4 = write_run(tmp_path, 4, [{"Sample": "C:/data/a.tif", "Value": 4}])
    preview_4.rename(tmp_path / "preview_4")

    dialog = RunComparisonDialog(start_path=tmp_path / "preview_4")

    assert Path(str(dialog.baseline_combo.currentData())).name == "preview_3"
    assert Path(str(dialog.current_combo.currentData())).name == "preview_4"
    assert dialog.comparison is not None
    assert dialog.comparison.measurement_changes[0].difference == 1


def test_windows_run_comparison_uses_qt_file_dialog_to_avoid_com_thread_errors(monkeypatch):
    monkeypatch.setattr(run_comparison_module.sys, "platform", "win32")

    options = run_comparison_module.file_dialog_options()

    assert options & QFileDialog.Option.DontUseNativeDialog
