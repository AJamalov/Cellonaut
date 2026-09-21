"""Dialog for comparing measurements and settings from completed runs or previews."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cellonaut.io.writers import write_dataframe_csv
from cellonaut.results.comparison import (
    ComparableRun,
    MeasurementChange,
    RunComparison,
    compare_runs,
    discover_comparable_runs,
    resolve_comparable_run,
)


# Qt's native Windows picker can repeatedly raise RPC_E_WRONG_THREAD in a
# process that also uses worker threads. The Qt picker avoids that COM path.
def file_dialog_options() -> QFileDialog.Option:
    if sys.platform.startswith("win"):
        return QFileDialog.Option.DontUseNativeDialog
    return QFileDialog.Option(0)


BASELINE_COLOR = QColor("#f3f3f3")
POSITIVE_COLOR = QColor("#4caf50")
NEGATIVE_COLOR = QColor("#d96c6c")


class SortableTableItem(QTableWidgetItem):
    """Table item that sorts by its underlying value instead of formatted text."""

    def __init__(self, text: str, sort_value: Any = None):
        super().__init__(text)
        self.sort_value = text.casefold() if sort_value is None else sort_value

    # Numeric columns need value-based sorting because formatted strings put 10 before 2.
    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, SortableTableItem):
            try:
                return self.sort_value < other.sort_value
            except TypeError:
                return str(self.sort_value).casefold() < str(other.sort_value).casefold()
        return super().__lt__(other)


class RunComparisonDialog(QDialog):
    """Compare exported summary measurements and recorded run settings."""

    # Run comparison is separate from the file browser so reviewing runs never changes
    # the folder or artifact currently open in the main window.
    def __init__(self, parent=None, *, start_path: Path | str = ""):
        super().__init__(parent)
        self.comparison: RunComparison | None = None
        self._visible_measurement_changes: list[MeasurementChange] = []

        self.setWindowTitle("Compare Runs")
        self.resize(1120, 720)
        self.setMinimumSize(820, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("Compare Runs")
        title.setProperty("uiRole", "dialogHeader")
        root.addWidget(title)

        hint = QLabel(
            "Choose an earlier run or preview as the baseline and a later one as the current result. "
            "Measurements are matched by sample and measurement name."
        )
        hint.setWordWrap(True)
        hint.setProperty("muted", "true")
        root.addWidget(hint)

        selector_row = QHBoxLayout()
        selector_row.setSpacing(8)
        selector_row.addWidget(QLabel("Baseline"))
        self.baseline_combo = QComboBox()
        self.baseline_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.baseline_combo.setMinimumContentsLength(18)
        selector_row.addWidget(self.baseline_combo, 1)
        self.baseline_browse_button = QPushButton("Browse...")
        self.baseline_browse_button.clicked.connect(lambda: self.browse_for_run(self.baseline_combo))
        selector_row.addWidget(self.baseline_browse_button)

        selector_row.addWidget(QLabel("Current"))
        self.current_combo = QComboBox()
        self.current_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.current_combo.setMinimumContentsLength(18)
        selector_row.addWidget(self.current_combo, 1)
        self.current_browse_button = QPushButton("Browse...")
        self.current_browse_button.clicked.connect(lambda: self.browse_for_run(self.current_combo))
        selector_row.addWidget(self.current_browse_button)

        self.compare_button = QPushButton("Compare")
        self.compare_button.setProperty("role", "primary")
        self.compare_button.clicked.connect(self.compare_selected_runs)
        selector_row.addWidget(self.compare_button)
        root.addLayout(selector_row)

        self.run_path_label = QLabel("Choose two completed run or preview folders.")
        self.run_path_label.setWordWrap(True)
        self.run_path_label.setProperty("muted", "true")
        root.addWidget(self.run_path_label)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self.build_measurements_tab()
        self.build_settings_tab()

        footer = QHBoxLayout()
        self.summary_label = QLabel("No comparison loaded.")
        self.summary_label.setWordWrap(True)
        footer.addWidget(self.summary_label, 1)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        footer.addWidget(self.close_button)
        root.addLayout(footer)

        self.populate_runs(start_path)

    def make_table(self, headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        return table

    # Filters belong immediately above the measurement table because they alter
    # only that view and do not affect settings differences.
    def build_measurements_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        self.measurement_filter = QLineEdit()
        self.measurement_filter.setPlaceholderText("Filter samples or measurements")
        self.measurement_filter.textChanged.connect(self.refresh_measurement_table)
        controls.addWidget(self.measurement_filter, 1)

        self.changed_only_checkbox = QCheckBox("Changed only")
        self.changed_only_checkbox.setChecked(True)
        self.changed_only_checkbox.toggled.connect(self.refresh_measurement_table)
        controls.addWidget(self.changed_only_checkbox)

        controls.addWidget(QLabel("Minimum change"))
        self.minimum_change_spin = QDoubleSpinBox()
        self.minimum_change_spin.setRange(0.0, 1_000_000.0)
        self.minimum_change_spin.setDecimals(2)
        self.minimum_change_spin.setSuffix(" %")
        self.minimum_change_spin.setToolTip(
            "Minimum absolute percentage change. Rows with N/A remain visible because their percentage cannot be calculated."
        )
        self.minimum_change_spin.valueChanged.connect(self.refresh_measurement_table)
        controls.addWidget(self.minimum_change_spin)

        self.export_measurements_button = QPushButton("Export CSV...")
        self.export_measurements_button.setEnabled(False)
        self.export_measurements_button.setToolTip("Export only rows matching the current search, Changed only, and Minimum change filters.")
        self.export_measurements_button.clicked.connect(self.export_measurement_comparison)
        controls.addWidget(self.export_measurements_button)
        layout.addLayout(controls)

        self.measurement_table = self.make_table(
            ["Sample", "Measurement", "Baseline", "Current", "Difference", "Change", "Status"]
        )
        comparison_note = QLabel(
            "Difference = current - baseline. Change = 100 ? difference / |baseline|. "
            "N/A means the baseline is zero or a value is missing (Added/Removed). "
            "Minimum change keeps N/A rows."
        )
        comparison_note.setWordWrap(True)
        comparison_note.setProperty("muted", "true")
        layout.addWidget(comparison_note)
        layout.addWidget(self.measurement_table, 1)
        self.tabs.addTab(tab, "Measurements")

    # Settings are listed separately because their before/after values are
    # categorical and a numeric percentage would imply meaning they do not have.
    def build_settings_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        note = QLabel("Only settings that differ between the two runs are shown.")
        note.setProperty("muted", "true")
        controls.addWidget(note, 1)
        self.export_settings_button = QPushButton("Export CSV...")
        self.export_settings_button.setToolTip("Export all differing settings between the two runs.")
        self.export_settings_button.setEnabled(False)
        self.export_settings_button.clicked.connect(self.export_settings_comparison)
        controls.addWidget(self.export_settings_button)
        layout.addLayout(controls)

        self.settings_table = self.make_table(["Setting", "Baseline", "Current", "Status"])
        layout.addWidget(self.settings_table, 1)
        self.tabs.addTab(tab, "Settings")

    # Discover sibling run_N and preview_N folders, then prefer the selected
    # operation type so opening from a preview does not default to full runs.
    def populate_runs(self, start_path: Path | str) -> None:
        runs = discover_comparable_runs(start_path) if str(start_path).strip() else []
        for run in runs:
            self.add_run_to_combo(self.baseline_combo, run)
            self.add_run_to_combo(self.current_combo, run)

        default_indices = self.default_comparison_indices(runs, start_path)
        if default_indices is not None:
            baseline_index, current_index = default_indices
            self.baseline_combo.setCurrentIndex(baseline_index)
            self.current_combo.setCurrentIndex(current_index)
            self.compare_selected_runs(show_errors=False)
        elif len(runs) == 1:
            self.baseline_combo.setCurrentIndex(0)
            self.current_combo.setCurrentIndex(0)

    # Use the selected operation as the current result when possible; otherwise
    # prefer two full runs before falling back to two previews or a mixed pair.
    def default_comparison_indices(
        self,
        runs: list[ComparableRun],
        start_path: Path | str,
    ) -> tuple[int, int] | None:
        if len(runs) < 2:
            return None

        selected_run: ComparableRun | None = None
        try:
            selected_run = resolve_comparable_run(start_path)
        except ValueError:
            pass

        if selected_run is not None and selected_run.operation_type in {"run", "preview"}:
            same_type = [index for index, run in enumerate(runs) if run.operation_type == selected_run.operation_type]
            selected_index = next(
                (
                    index
                    for index in same_type
                    if runs[index].operation_dir.resolve() == selected_run.operation_dir.resolve()
                ),
                None,
            )
            if selected_index is not None and len(same_type) >= 2:
                selected_position = same_type.index(selected_index)
                if selected_position > 0:
                    return same_type[selected_position - 1], selected_index
                return selected_index, same_type[1]

        for operation_type in ("run", "preview"):
            matching = [index for index, run in enumerate(runs) if run.operation_type == operation_type]
            if len(matching) >= 2:
                return matching[-2], matching[-1]
        return len(runs) - 2, len(runs) - 1

    def add_run_to_combo(self, combo: QComboBox, run: ComparableRun) -> int:
        path = str(run.operation_dir)
        for index in range(combo.count()):
            if str(combo.itemData(index)) == path:
                return index
        label = f"{run.label} | {run.operation_dir.parent}"
        combo.addItem(label, path)
        index = combo.count() - 1
        combo.setItemData(index, path, Qt.ItemDataRole.ToolTipRole)
        return index

    # Manual selection accepts a run, preview, or Results folder to match how
    # users naturally reach completed analyses in different file browsers.
    def browse_for_run(self, combo: QComboBox) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose completed run or preview",
            self.selector_start_folder(combo),
            options=file_dialog_options(),
        )
        if not selected:
            return
        try:
            run = resolve_comparable_run(selected)
        except ValueError as exc:
            QMessageBox.warning(self, "Compare Runs", str(exc))
            return
        index = self.add_run_to_combo(combo, run)
        combo.setCurrentIndex(index)

    def selector_start_folder(self, combo: QComboBox) -> str:
        path = str(combo.currentData() or "").strip()
        return path if path and Path(path).exists() else str(Path.home())

    # Compare on request or initial setup, rather than on every selector change.
    def compare_selected_runs(self, _checked: bool = False, *, show_errors: bool = True) -> None:
        baseline = str(self.baseline_combo.currentData() or "").strip()
        current = str(self.current_combo.currentData() or "").strip()
        if not baseline or not current:
            if show_errors:
                QMessageBox.information(self, "Compare Runs", "Choose a baseline and current run first.")
            return
        try:
            self.comparison = compare_runs(baseline, current)
        except (OSError, ValueError) as exc:
            if show_errors:
                QMessageBox.warning(self, "Compare Runs", str(exc))
            return

        self.run_path_label.setText(
            f"Baseline: {self.comparison.baseline_run.operation_dir}\n"
            f"Current: {self.comparison.current_run.operation_dir}"
        )
        self.export_measurements_button.setEnabled(True)
        self.export_settings_button.setEnabled(True)
        self.refresh_measurement_table()
        self.refresh_settings_table()
        self.refresh_summary()

    @staticmethod
    def format_number(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:.8g}"

    # Derive color from the signed numerical change, while treating additions
    # and removals as positive and negative structural changes respectively.
    @staticmethod
    def change_color(change: MeasurementChange) -> QColor:
        if change.status == "Added" or (change.difference is not None and change.difference > 0):
            return POSITIVE_COLOR
        if change.status == "Removed" or (change.difference is not None and change.difference < 0):
            return NEGATIVE_COLOR
        return BASELINE_COLOR

    # Additions and zero-baseline changes remain visible when a percentage filter
    # is active because no meaningful percentage can be calculated for them.
    def filtered_measurement_changes(self) -> list[MeasurementChange]:
        if self.comparison is None:
            return []
        query = self.measurement_filter.text().strip().casefold()
        changed_only = self.changed_only_checkbox.isChecked()
        minimum_change = self.minimum_change_spin.value()
        visible: list[MeasurementChange] = []
        for change in self.comparison.measurement_changes:
            if changed_only and change.status == "Unchanged":
                continue
            if query and query not in f"{change.sample} {change.measurement} {change.measurement_key}".casefold():
                continue
            if change.percent_change is not None and abs(change.percent_change) < minimum_change:
                continue
            visible.append(change)
        return visible

    # Color values rather than table backgrounds so selected rows remain legible
    # and the baseline stays a stable white reference throughout the comparison.
    def refresh_measurement_table(self, *_args) -> None:
        self._visible_measurement_changes = self.filtered_measurement_changes()
        table = self.measurement_table
        table.setSortingEnabled(False)
        table.setRowCount(len(self._visible_measurement_changes))
        for row, change in enumerate(self._visible_measurement_changes):
            color = self.change_color(change)
            percent_text = "N/A" if change.percent_change is None else f"{change.percent_change:+.2f}%"
            difference_text = self.format_number(change.difference)
            if change.difference is not None and change.difference > 0:
                difference_text = f"+{difference_text}"
            values = [
                SortableTableItem(change.sample),
                SortableTableItem(change.measurement),
                SortableTableItem(
                    self.format_number(change.baseline),
                    change.baseline if change.baseline is not None else float("-inf"),
                ),
                SortableTableItem(
                    self.format_number(change.current), change.current if change.current is not None else float("-inf")
                ),
                SortableTableItem(
                    difference_text, change.difference if change.difference is not None else float("-inf")
                ),
                SortableTableItem(
                    percent_text, change.percent_change if change.percent_change is not None else float("-inf")
                ),
                SortableTableItem(change.status),
            ]
            values[2].setForeground(BASELINE_COLOR)
            for item in values[3:]:
                item.setForeground(color)
            for column, item in enumerate(values):
                table.setItem(row, column, item)
        table.setSortingEnabled(True)

    # Settings use white for ordinary replacements; only added and removed
    # entries carry directional color because categorical changes have no sign.
    def refresh_settings_table(self) -> None:
        changes = self.comparison.setting_changes if self.comparison is not None else []
        table = self.settings_table
        table.setSortingEnabled(False)
        table.setRowCount(len(changes))
        for row, change in enumerate(changes):
            current_color = (
                POSITIVE_COLOR
                if change.status == "Added"
                else NEGATIVE_COLOR
                if change.status == "Removed"
                else BASELINE_COLOR
            )
            values = [
                SortableTableItem(change.setting),
                SortableTableItem(change.baseline),
                SortableTableItem(change.current),
                SortableTableItem(change.status),
            ]
            values[1].setForeground(BASELINE_COLOR)
            values[2].setForeground(current_color)
            values[3].setForeground(current_color)
            for column, item in enumerate(values):
                table.setItem(row, column, item)
        table.setSortingEnabled(True)

    # Report unchanged values as well so filtered tables cannot make a sparse comparison look complete.
    def refresh_summary(self) -> None:
        if self.comparison is None:
            self.summary_label.setText("No comparison loaded.")
            return
        changes = self.comparison.measurement_changes
        changed = sum(change.status == "Changed" for change in changes)
        added = sum(change.status == "Added" for change in changes)
        removed = sum(change.status == "Removed" for change in changes)
        unchanged = sum(change.status == "Unchanged" for change in changes)
        self.summary_label.setText(
            f"Measurements: {changed} changed, {added} added, {removed} removed, {unchanged} unchanged. "
            f"Settings: {len(self.comparison.setting_changes)} changed."
        )

    # Export only the measurements included by the current filters.
    def export_measurement_comparison(self) -> None:
        if self.comparison is None:
            return
        rows = [
            {
                "Sample": change.sample,
                "Measurement": change.measurement,
                "Measurement key": change.measurement_key,
                "Baseline": change.baseline,
                "Current": change.current,
                "Difference": change.difference,
                "Percent change": change.percent_change,
                "Status": change.status,
            }
            for change in self._visible_measurement_changes
        ]
        self._export_comparison_csv("measurement", "Run_Comparison.csv", rows)

    # Export settings separately from measurement changes.
    def export_settings_comparison(self) -> None:
        if self.comparison is None:
            return
        rows = [
            {
                "Setting": change.setting,
                "Baseline": change.baseline,
                "Current": change.current,
                "Status": change.status,
            }
            for change in self.comparison.setting_changes
        ]
        self._export_comparison_csv("settings", "Run_Settings_Comparison.csv", rows)

    # Keep extension handling and write failures identical for both reports.
    def _export_comparison_csv(self, report: str, filename: str, rows: list[dict[str, Any]]) -> None:
        if self.comparison is None:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            f"Export {report} comparison",
            str(self.comparison.current_run.operation_dir / filename),
            "CSV files (*.csv)",
            options=file_dialog_options(),
        )
        if not path:
            return
        output = Path(path)
        if output.suffix.casefold() != ".csv":
            output = output.with_suffix(".csv")
        try:
            write_dataframe_csv(pd.DataFrame(rows), output, index=False)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Export comparison", f"Could not save the comparison:\n{exc}")
