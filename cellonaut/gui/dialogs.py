"""Small dialogs used by setup checks, filters, measurements, and help flows.

These dialogs collect focused user choices without owning pipeline state. The
main window passes current labels/settings in, and reads normalized values back
when the user accepts.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def count_label(count: int, singular: str, plural: str | None = None) -> str:
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {noun}"


class TextReportDialog(QDialog):
    """Simple scrollable text report dialog."""

    # Keep plain-text reports in a dedicated read-only viewer so long diagnostics
    # remain searchable and copyable without mixing them into editable fields.
    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 700)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setFont(QFont("Consolas", 10))
        self.text_box.setPlainText(text or "")

        root.addWidget(self.text_box, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        self.copy_button = QPushButton("Copy")
        self.close_button = QPushButton("Close")

        self.copy_button.clicked.connect(self.copy_text)
        self.close_button.clicked.connect(self.accept)

        button_row.addWidget(self.copy_button)
        button_row.addWidget(self.close_button)

        root.addLayout(button_row)

    def copy_text(self):
        QApplication.clipboard().setText(self.text_box.toPlainText())


class ConfigurationSummaryDialog(QDialog):
    """Structured setup report shown before analysis."""

    def __init__(self, summary: dict, raw_text: str = "", parent=None):
        super().__init__(parent)
        self.summary = summary or {}
        self.raw_text = raw_text or ""

        self.setWindowTitle("Check Setup")
        self.resize(1050, 760)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QLabel("Check Setup")
        header.setProperty("uiRole", "dialogHeader")
        root.addWidget(header)

        subtitle = QLabel(
            "Check folders, channels, masks, Cellpose, cell groups, "
            "measurements, and warnings before launching the analysis."
        )
        subtitle.setWordWrap(True)
        subtitle.setProperty("muted", "true")
        root.addWidget(subtitle)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)

        warnings = self.summary.get("warnings", []) or []
        setup_checks = self.summary.get("setup_checks", []) or []
        relationships = self.summary.get("relationships", []) or []
        images = self.summary.get("images", []) or []
        dry_run = self.summary.get("dry_run", {}) or {}
        blocked_checks = [item for item in setup_checks if str(item.get("status", "")).upper() == "BLOCKED"]
        warning_checks = [item for item in setup_checks if str(item.get("status", "")).upper() == "WARNING"]

        if blocked_checks:
            status_row.addWidget(
                self.make_status_pill(
                    count_label(len(blocked_checks), "blocker"),
                    "danger",
                )
            )
        elif warning_checks:
            status_row.addWidget(
                self.make_status_pill(
                    count_label(len(warning_checks), "setup warning"),
                    "warning",
                )
            )
        elif setup_checks:
            status_row.addWidget(
                self.make_status_pill(
                    "Setup checks OK",
                    "success",
                )
            )

        status_row.addWidget(
            self.make_status_pill(
                count_label(len(images), "channel/mask item"),
                "neutral",
            )
        )
        status_row.addWidget(
            self.make_status_pill(
                count_label(len(relationships), "mask relationship"),
                "success" if relationships else "warning",
            )
        )

        if dry_run.get("available", False):
            ready = dry_run.get("ready", 0)
            total = dry_run.get("total", 0)
            status_row.addWidget(
                self.make_status_pill(
                    f"{ready}/{total} samples ready",
                    "success" if ready == total and total > 0 else "warning",
                )
            )
        else:
            status_row.addWidget(
                self.make_status_pill(
                    "Sample check unavailable",
                    "warning",
                )
            )

        status_row.addWidget(
            self.make_status_pill(
                "No warnings" if not warnings else count_label(len(warnings), "warning"),
                "success" if not warnings else "warning",
            )
        )
        status_row.addStretch(1)
        root.addLayout(status_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(scroll, 1)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(10)

        layout.addWidget(self.build_setup_checks_card())
        layout.addWidget(self.build_folders_card())
        layout.addWidget(self.build_images_card())
        layout.addWidget(self.build_relationships_card())
        layout.addWidget(self.build_cellpose_card())
        layout.addWidget(self.build_filters_card())
        layout.addWidget(self.build_measurements_card())
        layout.addWidget(self.build_dry_run_card())
        layout.addWidget(self.build_warnings_card())
        layout.addStretch(1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        self.copy_button = QPushButton("Copy summary")
        self.copy_button.clicked.connect(self.copy_text_summary)

        self.raw_button = QPushButton("Open text report")
        self.raw_button.clicked.connect(self.open_raw_text)

        self.close_button = QPushButton("Close")
        self.close_button.setProperty("role", "primary")
        self.close_button.clicked.connect(self.accept)

        button_row.addWidget(self.copy_button)
        button_row.addWidget(self.raw_button)
        button_row.addWidget(self.close_button)
        root.addLayout(button_row)

    # Express severity through theme properties so status colors stay readable
    # in every supported appearance mode.
    def make_status_pill(self, text: str, kind: str = "neutral") -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setProperty("uiRole", "statusPill")
        label.setProperty("statusKind", kind)
        return label

    def make_card(self, title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setProperty("card", "true")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setProperty("uiRole", "cardTitle")
        layout.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setWordWrap(True)
            subtitle_label.setProperty("muted", "true")
            layout.addWidget(subtitle_label)

        return card, layout

    def make_kv_table(self, rows: list[tuple[str, str]]) -> QTableWidget:
        return self.make_table(["Setting", "Value"], [[key, value] for key, value in rows], max_height=260)

    # Stretch the descriptive final column while keeping identifiers compact,
    # which makes wide setup details readable without horizontal scrolling.
    def make_table(self, headers: list[str], rows: list[list[str]], max_height: int = 320) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setWordWrap(True)
        table.setTextElideMode(Qt.TextElideMode.ElideNone)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        for col in range(len(headers)):
            mode = (
                QHeaderView.ResizeMode.Stretch if col == len(headers) - 1 else QHeaderView.ResizeMode.ResizeToContents
            )
            table.horizontalHeader().setSectionResizeMode(col, mode)

        for row_idx, row_values in enumerate(rows):
            for col_idx, value in enumerate(row_values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row_idx, col_idx, item)

        table.setMinimumHeight(min(max_height, 42 + max(1, len(rows)) * 30))
        return table

    # Present paths together because they are usually checked as a group when a
    # setup works on one machine but not another.
    def build_folders_card(self) -> QFrame:
        card, layout = self.make_card("Folders", "Where Cellonaut will read data from and write results.")
        folders = self.summary.get("folders", {}) or {}

        rows = [
            ("Input folder", folders.get("input", "")),
            ("Output folder", folders.get("output", "")),
            ("Bundled Fiji runtime", folders.get("fiji", "")),
            ("Input layout", folders.get("layout", "")),
        ]

        layout.addWidget(self.make_kv_table(rows))
        return card

    # Put blocking checks first in the report so a user can resolve run-stopping
    # conditions before reviewing lower-priority configuration details.
    def build_setup_checks_card(self) -> QFrame:
        card, layout = self.make_card(
            "Setup checks",
            "Combined validation, measurement warnings, and sample readiness.",
        )

        setup_checks = self.summary.get("setup_checks", []) or []
        if not setup_checks:
            empty = QLabel("No setup checks were generated.")
            empty.setProperty("muted", "true")
            layout.addWidget(empty)
            return card

        rows = [
            [
                item.get("status", ""),
                item.get("check", ""),
                item.get("detail", ""),
            ]
            for item in setup_checks
        ]

        layout.addWidget(
            self.make_table(
                ["Status", "Check", "Detail"],
                rows,
                max_height=360,
            )
        )
        return card

    # Show channels and mask sources in one table because their association is
    # more important during setup review than their internal definition type.
    def build_images_card(self) -> QFrame:
        card, layout = self.make_card(
            "Channels and masks",
            "Detected channels and configured mask sources.",
        )

        images = self.summary.get("images", []) or []
        rows = []

        for img in images:
            rows.append(
                [
                    img.get("type", ""),
                    img.get("name", ""),
                    img.get("folder", ""),
                    img.get("mask_source", "") or "-",
                    img.get("probability_class", ""),
                    img.get("threshold", ""),
                    img.get("classifier_name", "") or "(none)",
                ]
            )

        if not rows:
            rows = [["(none)", "", "", "", "", "", ""]]

        layout.addWidget(
            self.make_table(
                [
                    "Type",
                    "Name",
                    "Source",
                    "Mask source",
                    "Probability-map class",
                    "Auto Threshold method",
                    "Weka classifier",
                ],
                rows,
                max_height=360,
            )
        )
        return card

    # Translate matrix entries into plain language so users can verify which
    # mask defines each measured region without decoding ON/OFF coordinates.
    def build_relationships_card(self) -> QFrame:
        card, layout = self.make_card(
            "Measurement regions",
            "Which channels are measured, and which masks are applied to them.",
        )

        relationships = self.summary.get("relationships", []) or []

        if not relationships:
            empty = QLabel(
                "No configured-mask or Cellpose-mask measurement regions are selected."
            )
            empty.setProperty("muted", "true")
            layout.addWidget(empty)
            return card

        rows = []
        for rel in relationships:
            if rel.get("kind") == "cellpose":
                meaning = f"Measured using reusable {rel.get('target', '')} mask"
            elif rel.get("self", False):
                meaning = "Measured with own mask"
            else:
                meaning = f"Measured using {rel.get('target', '')} mask"

            rows.append(
                [
                    rel.get("source", ""),
                    rel.get("target", ""),
                    meaning,
                ]
            )

        layout.addWidget(
            self.make_table(
                ["Measured channel", "Mask", "Meaning"],
                rows,
                max_height=360,
            )
        )
        return card

    # Omit an empty technical table when Cellpose is unused while still making
    # that intentional state explicit in the setup report.
    def build_cellpose_card(self) -> QFrame:
        card, layout = self.make_card(
            "Cellpose",
            "Cellpose settings used to create whole-cell masks.",
        )

        rows = self.summary.get("cell_segmentation", []) or []

        if not rows:
            empty = QLabel("No reusable Cellpose masks are configured.")
            empty.setProperty("muted", "true")
            layout.addWidget(empty)
            return card

        table_rows = []
        for item in rows:
            table_rows.append(
                [
                    f"{item.get('source', '')}_Cellpose",
                    item.get("seg_source", ""),
                    item.get("diameter", ""),
                    item.get("min_size", ""),
                    item.get("cellprob", ""),
                    item.get("flow", ""),
                    "Yes" if item.get("remove_border", True) else "No",
                ]
            )

        layout.addWidget(
            self.make_table(
                [
                    "Cellpose mask",
                    "Cellpose source channel",
                    "Diameter (px)",
                    "Minimum cell area (px²)",
                    "Probability threshold",
                    "Flow threshold",
                    "Remove border",
                ],
                table_rows,
                max_height=300,
            )
        )
        return card

    # Show both condition categories together so the group match rule is clear.
    def build_filters_card(self) -> QFrame:
        card, layout = self.make_card(
            "Cell Groups",
            "Cell and target-mask conditions used to define groups for preview and filtered CSV exports.",
        )

        filters = self.summary.get("filters", []) or []

        if not filters:
            empty = QLabel("No cell-group conditions configured.")
            empty.setProperty("muted", "true")
            layout.addWidget(empty)
            return card

        rows = []
        for item in filters:
            rows.append(
                [
                    item.get("source", ""),
                    item.get("cell_limits", "") or "(none)",
                    item.get("mask_limits", "") or "(none)",
                    item.get("mode", ""),
                    item.get("mask_intensity_source", ""),
                    "Yes" if item.get("exclude", False) else "No",
                ]
            )

        layout.addWidget(
            self.make_table(
                [
                    "Measured channel",
                    "Cell conditions",
                    "Target-mask conditions",
                    "Match rule",
                    "Intensity source",
                    "Exclude group from CSV",
                ],
                rows,
                max_height=260,
            )
        )
        return card

    # Summarize shared measurement choices instead of repeating the same list
    # once for every measured channel.
    def build_measurements_card(self) -> QFrame:
        card, layout = self.make_card(
            "Measurement options",
            "Shared measurement metrics for all measured channels. Necessary results are saved automatically.",
        )

        measurements = self.summary.get("measurements", []) or []

        if not measurements:
            empty = QLabel("No measurement settings found.")
            empty.setProperty("muted", "true")
            layout.addWidget(empty)
            return card

        rows = []
        for item in measurements:
            enabled = item.get("enabled", []) or []
            rows.append(
                [
                    item.get("source", ""),
                    str(len(enabled)),
                    ", ".join(enabled[:12]) + (f", +{len(enabled) - 12} more" if len(enabled) > 12 else ""),
                ]
            )

        layout.addWidget(
            self.make_table(
                ["Applies to", "Enabled count", "Enabled measurements"],
                rows,
                max_height=300,
            )
        )
        return card

    # Keep sample readiness separate from static setup checks because it reports
    # what was actually discovered in the selected input folder.
    def build_dry_run_card(self) -> QFrame:
        card, layout = self.make_card(
            "Sample detection",
            "Checks the input folder before starting Fiji, mask generation, or Cellpose.",
        )

        dry_run = self.summary.get("dry_run", {}) or {}

        if not dry_run.get("available", False):
            warning = QLabel(dry_run.get("error", "Sample detection information is unavailable."))
            warning.setWordWrap(True)
            warning.setProperty("messageState", "warning")
            layout.addWidget(warning)
            return card

        rows = [
            ("Detected samples", dry_run.get("total", 0)),
            ("Ready samples", dry_run.get("ready", 0)),
            ("Incomplete samples", dry_run.get("incomplete", 0)),
            ("Samples with warnings", dry_run.get("warning_count", 0)),
        ]

        layout.addWidget(self.make_kv_table([(str(k), str(v)) for k, v in rows]))

        problems = dry_run.get("problems", []) or []
        if problems:
            problem_label = QLabel("First detected problems:")
            problem_label.setProperty("uiRole", "sectionTitle")
            layout.addWidget(problem_label)

            problem_rows = []
            for entry in problems[:20]:
                problem_rows.append(
                    [
                        entry.get("sample", ""),
                        ", ".join(entry.get("missing_required", []) or []) or "-",
                        ", ".join(entry.get("missing_optional", []) or []) or "-",
                        ", ".join(entry.get("ambiguous", []) or []) or "-",
                        ", ".join(entry.get("reuse_mask_warnings", []) or []) or "-",
                    ]
                )

            layout.addWidget(
                self.make_table(
                    ["Sample", "Missing input", "Missing optional input", "Ambiguous", "Mask reuse warning"],
                    problem_rows,
                    max_height=320,
                )
            )

        return card

    # Group warning messages under one heading so severity is clear without a
    # repeated prefix on every line.
    def build_warnings_card(self) -> QFrame:
        warnings = self.summary.get("warnings", []) or []
        card, layout = self.make_card(
            "Warnings",
            "Potential problems detected in the current setup.",
        )

        if not warnings:
            ok = QLabel("No configuration warnings found.")
            ok.setProperty("messageState", "ok")
            layout.addWidget(ok)
            return card

        for warning in warnings:
            label = QLabel(str(warning))
            label.setWordWrap(True)
            label.setProperty("messageState", "warning")
            layout.addWidget(label)

        return card

    def copy_text_summary(self):
        QApplication.clipboard().setText(self.raw_text)

    # Open a separate viewer rather than replacing the structured report, which
    # lets users compare readable cards with their portable text equivalent.
    def open_raw_text(self):
        dialog = TextReportDialog("Setup Check Text Report", self.raw_text, self)
        dialog.exec()


class PresetInfoDialog(QDialog):
    """Dialog for entering a preset name."""

    def __init__(self, name: str = "", parent=None):
        super().__init__(parent)
        updating = bool(str(name or "").strip())
        self.setWindowTitle("Save Preset")
        self.resize(460, 170)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel("Save Preset")
        title.setStyleSheet("font-size: 13pt; font-weight: 700;")
        root.addWidget(title)

        name_label = QLabel("Preset name")
        root.addWidget(name_label)

        self.name_edit = QLineEdit(str(name or ""))
        self.name_edit.setPlaceholderText("Example: Golgi analysis")
        root.addWidget(self.name_edit)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        self.ok_button = QPushButton("Save" if updating else "Save Preset")
        self.cancel_button = QPushButton("Cancel")
        self.ok_button.setProperty("role", "primary")
        self.ok_button.setDefault(True)
        self.ok_button.setEnabled(bool(self.name_edit.text().strip()))

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.name_edit.textChanged.connect(lambda text: self.ok_button.setEnabled(bool(text.strip())))

        button_row.addWidget(self.ok_button)
        button_row.addWidget(self.cancel_button)
        root.addLayout(button_row)

    def get_name(self) -> str:
        return self.name_edit.text().strip()
