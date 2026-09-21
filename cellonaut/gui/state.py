"""Typed runtime state owned by the composed Qt main window."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PySide6.QtWidgets import QComboBox, QLabel, QListWidget, QWidget

from cellonaut.gui.widgets import EntryRow, PathRow


@dataclass(slots=True)
class WorkerSlot:
    """Keep one worker and its owning thread together throughout their lifecycle."""

    worker: Any | None = None
    thread: Any | None = None

    @property
    def active(self) -> bool:
        return self.worker is not None

    def clear(self) -> None:
        self.worker = None
        self.thread = None


@dataclass(slots=True)
class GuiTaskState:
    """Track pipeline, preview, and ND2 tasks; only one runs at a time."""

    pipeline: WorkerSlot = field(default_factory=WorkerSlot)
    preview: WorkerSlot = field(default_factory=WorkerSlot)
    nd2: WorkerSlot = field(default_factory=WorkerSlot)
    running: bool = False
    preview_was_cancelled: bool = False
    close_retry_scheduled: bool = False

    def active_workers(self) -> list[tuple[Any, str, str]]:
        candidates = (
            (self.pipeline, "RUN", "run"),
            (self.preview, "PREVIEW", "preview"),
            (self.nd2, "ND2", "ND2 conversion"),
        )
        return [(slot.worker, scope, label) for slot, scope, label in candidates if slot.worker is not None]


@dataclass(slots=True)
class PreviewState:
    """Authoritative data for the open Preview, initialized before UI construction.

    Qt widgets, scene items, view transforms and timers remain with the view.
    Run configuration and worker lifetimes belong to their existing owners.
    """

    pages: list[Any] = field(default_factory=list)
    page_index: int = 0
    file_path: Any | None = None
    overlay_colors: list[Any] = field(default_factory=list)
    overlay_opacities: list[Any] = field(default_factory=list)
    current_layer_roles: list[Any] = field(default_factory=list)
    render_revision: int = 0
    page_revisions: list[Any] = field(default_factory=list)
    normalized_page_cache: dict[Any, Any] = field(default_factory=dict)
    filter_data: Any | None = None
    filter_excluded_labels: set[Any] = field(default_factory=set)
    filter_visible: bool = True
    mask_adjustments_by_target: dict[str, Any] = field(default_factory=dict)
    tiff_model: Any | None = None
    page_labels: list[Any] = field(default_factory=list)
    current_labels: list[Any] = field(default_factory=list)
    current_layer_keys: list[str] = field(default_factory=list)
    layer_order: list[int] = field(default_factory=list)
    layer_visibility: list[bool] = field(default_factory=list)
    selected_layer_index: int | None = None
    composite_mode: bool = False
    force_additive_composite: bool = False
    preserve_layer_colors: bool = False
    rendered_page_order: list[int] = field(default_factory=list)
    group_layer_display_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    cell_group_result: Any | None = None
    mask_adjustment_last_preview_values: dict[str, Any] = field(default_factory=dict)
    mask_adjust_target: str = ""
    tools_source_label_override: str = ""
    tools_source_matches_pipeline: bool = True
    artifact_mode: str = ""
    artifact_entries: list[Any] = field(default_factory=list)
    artifact_index: int = -1
    artifact_file_index: int = -1
    artifact_results_root: Any | None = None


@dataclass(slots=True)
class ImageDefinitionRowWidgets:
    """Widgets and tab indexes belonging to one editable image definition."""

    definition_index: int
    default_name: str
    name: EntryRow | None = None
    classifier: PathRow | None = None
    mask_source_mode: QComboBox | None = None
    mask_source_channel: QComboBox | None = None
    probability_class_index: EntryRow | None = None
    threshold_method: QComboBox | None = None
    threshold_method_host: QWidget | None = None
    combined_mask_sources: QListWidget | None = None
    combined_mask_operation: QComboBox | None = None
    roi_status: QLabel | None = None
    channel_tab: QWidget | None = None
    mask_tab: QWidget | None = None
    channel_tab_index: int = -1
    mask_tab_index: int = -1
